import math
import os
import re
import gc
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "exports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO = {"mp4", "mov", "m4v", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}
ALLOWED_AUDIO = {"mp3", "wav", "m4a", "aac", "ogg"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
# The visual reference itself is 576x1024. Matching it also keeps peak memory
# safely below Render Free's limit while retaining a crisp 9:16 social export.
OUTPUT_WIDTH = 576
OUTPUT_HEIGHT = 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
RENDER_QUEUE = ThreadPoolExecutor(max_workers=1)
SUBJECT_NET = None
SUBJECT_NET_FAILED = False
SUBJECT_NET_LOCK = threading.Lock()
MODEL_DIR = Path(os.environ.get("PETCUT_MODEL_DIR", BASE_DIR / "data" / "models"))
YOLO_CFG = MODEL_DIR / "yolov4-tiny.cfg"
YOLO_WEIGHTS = MODEL_DIR / "yolov4-tiny.weights"
YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
SUBJECT_CLASSES = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def extension_allowed(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True)


def media_duration(path: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)])
    return float(result.stdout.strip())


def analyze_rhythm(audio_path: Path) -> dict:
    wav_path = audio_path.with_suffix(".analysis.wav")
    try:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "22050", str(wav_path)])
        with wave.open(str(wav_path), "rb") as source:
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float32)
            rate = source.getframerate()
        if len(samples) < rate * 3:
            return {"bpm": 120, "edit_bpm": 120, "onsets": []}
        hop = 1024
        frames = samples[: len(samples) // hop * hop].reshape(-1, hop)
        energy = np.maximum(np.sqrt(np.mean(frames**2, axis=1)) - np.mean(np.sqrt(np.mean(frames**2, axis=1))), 0)
        if not np.any(energy):
            return {"bpm": 120, "edit_bpm": 120, "onsets": []}
        correlation = np.correlate(energy, energy, mode="full")[len(energy) - 1 :]
        low, high = max(1, round((60 / 190) * rate / hop)), min(len(correlation), round((60 / 70) * rate / hop))
        bpm = max(70, min(190, round(60 * rate / (hop * (low + int(np.argmax(correlation[low:high])))))))

        # Fast social edits usually cut at double-time when a track is detected
        # around 70-100 BPM. Keep the musical BPM and the edit BPM separate.
        edit_bpm = bpm
        while edit_bpm < 108:
            edit_bpm *= 2
        while edit_bpm > 190:
            edit_bpm = round(edit_bpm / 2)

        # Lightweight onset detection used to align flashes and scene changes.
        window = np.hanning(hop).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
        flux = np.zeros(len(spectrum), dtype=np.float32)
        flux[1:] = np.maximum(spectrum[1:] - spectrum[:-1], 0).sum(axis=1)
        if np.max(flux) > 0:
            flux /= np.max(flux)
        onsets = []
        minimum_gap = max(2, round(0.18 * rate / hop))
        last_peak = -minimum_gap
        for index in range(2, len(flux) - 2):
            local = flux[max(0, index - 8) : min(len(flux), index + 9)]
            threshold = float(np.median(local) + 1.5 * np.median(np.abs(local - np.median(local))))
            if flux[index] >= max(0.08, threshold) and flux[index] == np.max(flux[index - 2 : index + 3]) and index - last_peak >= minimum_gap:
                onsets.append(round(index * hop / rate, 3))
                last_peak = index
        # Refine the visual tempo from the detected attacks. Short gaps are
        # half-beats and long gaps are two beats; normalising them makes the
        # cut grid follow the actual song instead of only an autocorrelation.
        onset_gaps = []
        for first, second in zip(onsets, onsets[1:]):
            gap = second - first
            if 0.15 <= gap < 0.30:
                gap *= 2
            elif 0.65 <= gap <= 1.30:
                gap /= 2
            if 0.30 <= gap <= 0.60:
                onset_gaps.append(gap)
        if len(onset_gaps) >= 5:
            edit_bpm = max(108, min(190, round(60 / float(np.median(onset_gaps)))))
        return {"bpm": bpm, "edit_bpm": int(edit_bpm), "onsets": onsets}
    except Exception:
        return {"bpm": 120, "edit_bpm": 120, "onsets": []}
    finally:
        wav_path.unlink(missing_ok=True)


def estimate_bpm(audio_path: Path) -> int:
    return analyze_rhythm(audio_path)["edit_bpm"]


def recommended_scenes(duration: float, bpm: int, style: str) -> int:
    beats_per_scene = {"cinematic": 4, "beat": 2, "collage": 2, "floating": 1}.get(style, 3)
    return max(3, min(18, math.ceil(duration / ((60 / bpm) * beats_per_scene))))


def recommended_content(duration: float, style: str) -> int:
    if style == "floating":
        return max(6, min(14, math.ceil(duration / 1.5)))
    return max(3, min(12, math.ceil(duration / 2.5)))


def final_filter(style: str, title: str) -> str:
    saturation = {"cinematic": "1.18", "beat": "1.35", "collage": "1.25", "floating": "1.12"}[style]
    filters = [
        f"eq=contrast=1.10:saturation={saturation}:brightness=0.015",
        "unsharp=5:5:0.55:5:5:0.0",
    ]
    if style in {"collage", "floating"}:
        filters.append("vignette=PI/5")
    clean_title = re.sub(r"[^A-Za-z0-9 !?-]", "", title)[:32]
    if clean_title:
        filters.append("drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                       f"text='{clean_title}':fontcolor=white:fontsize=72:borderw=4:bordercolor=black:"
                       "x=(w-text_w)/2:y=h*0.15:enable='between(t,0,2.8)'")
    return ",".join(filters)


def scene_filter(index: int, style: str, bpm: int, duration: float) -> str:
    """A distinct punchy camera move for every beat-sized scene."""
    strength = {"cinematic": 42, "beat": 105, "collage": 75, "floating": 65}[style]
    direction = index % 4
    if direction == 0:
        position = f"x='(in_w-out_w)/2+{strength}*t/{duration:.3f}':y='(in_h-out_h)/2'"
    elif direction == 1:
        position = f"x='(in_w-out_w)/2-{strength}*t/{duration:.3f}':y='(in_h-out_h)/2'"
    elif direction == 2:
        position = f"x='(in_w-out_w)/2':y='(in_h-out_h)/2+{strength}*t/{duration:.3f}'"
    else:
        position = f"x='(in_w-out_w)/2':y='(in_h-out_h)/2-{strength}*t/{duration:.3f}'"
    flash = "drawbox=color=white@0.72:t=fill:enable='between(t\\,0\\,0.045)'" if index else "null"
    contrast = "1.18" if style == "beat" and index % 2 else "1.08"
    return (
        f"scale=900:1600:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:{position},"
        f"eq=contrast={contrast}:saturation={'1.28' if style == 'beat' else '1.14'},fps=30,{flash}"
    )


def source_frame(path: Path, frame_number: int, frame_count: int) -> Image.Image:
    if path.suffix[1:].lower() in ALLOWED_IMAGE:
        frame = Image.open(path).convert("RGBA")
        frame.thumbnail((720, 1280), Image.Resampling.LANCZOS)
        return frame
    length = media_duration(path)
    timestamp = max(0.08, min(length - 0.08, length * (frame_number + 1) / (frame_count + 1)))
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    frame = Image.open(BytesIO(result.stdout)).convert("RGBA")
    frame.thumbnail((720, 1280), Image.Resampling.LANCZOS)
    return frame


def subject_detector():
    global SUBJECT_NET, SUBJECT_NET_FAILED
    if SUBJECT_NET is not None:
        return SUBJECT_NET
    if SUBJECT_NET_FAILED:
        return None
    with SUBJECT_NET_LOCK:
        if SUBJECT_NET is not None:
            return SUBJECT_NET
        if SUBJECT_NET_FAILED:
            return None
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            for path, url, minimum_size in (
                (YOLO_CFG, YOLO_CFG_URL, 2_000),
                (YOLO_WEIGHTS, YOLO_WEIGHTS_URL, 20_000_000),
            ):
                if path.exists() and path.stat().st_size < minimum_size:
                    path.unlink()
                if not path.exists():
                    temporary = path.with_suffix(path.suffix + ".part")
                    temporary.unlink(missing_ok=True)
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    if temporary.stat().st_size < minimum_size:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError("Modello soggetto incompleto")
                    temporary.replace(path)
            SUBJECT_NET = cv2.dnn.readNetFromDarknet(str(YOLO_CFG), str(YOLO_WEIGHTS))
        except Exception:
            SUBJECT_NET_FAILED = True
            app.logger.exception("Subject detector unavailable; using centred fallback")
    return SUBJECT_NET


def detect_subject(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find a person/animal box; fall back to a central portrait box."""
    height, width = rgb.shape[:2]
    try:
        network = subject_detector()
        if network is None:
            return int(width * 0.12), int(height * 0.08), int(width * 0.76), int(height * 0.86)
        blob = cv2.dnn.blobFromImage(rgb, 1 / 255.0, (416, 416), swapRB=False, crop=False)
        with SUBJECT_NET_LOCK:
            network.setInput(blob)
            outputs = network.forward(network.getUnconnectedOutLayersNames())
        candidates = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(detection[4] * scores[class_id])
                if class_id not in SUBJECT_CLASSES or confidence < 0.10:
                    continue
                center_x, center_y, box_width, box_height = detection[:4] * np.array([width, height, width, height])
                x = max(0, int(center_x - box_width / 2))
                y = max(0, int(center_y - box_height / 2))
                box_width = min(width - x, int(box_width))
                box_height = min(height - y, int(box_height))
                centrality = 1 - min(1, abs(center_x - width / 2) / width + abs(center_y - height / 2) / height)
                candidates.append((confidence + centrality * 0.10, (x, y, box_width, box_height)))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
    except Exception:
        app.logger.exception("Subject detection failed; using centred fallback")
    return int(width * 0.12), int(height * 0.08), int(width * 0.76), int(height * 0.86)


def release_subject_detector():
    global SUBJECT_NET
    with SUBJECT_NET_LOCK:
        SUBJECT_NET = None
    gc.collect()


def isolate_subject(frame: Image.Image, motion: np.ndarray | None = None, stability: np.ndarray | None = None, subject_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Detector and temporal-guided GrabCut with a conservative colour trimap."""
    rgb = np.array(frame.convert("RGB"))
    height, width = rgb.shape[:2]
    x, y, box_width, box_height = subject_box or detect_subject(rgb)
    x, y = max(0, x), max(0, y)
    box_width = max(2, min(width - x, box_width))
    box_height = max(2, min(height - y, box_height))

    yy, xx = np.mgrid[:height, :width]
    center_x, center_y = x + box_width * 0.5, y + box_height * 0.43
    subject_zone = ((xx - center_x) / max(2, box_width * 0.48)) ** 2 + ((yy - center_y) / max(2, box_height * 0.62)) ** 2 < 1
    core = ((xx - center_x) / max(2, box_width * 0.19)) ** 2 + ((yy - center_y) / max(2, box_height * 0.29)) ** 2 < 1
    protected_zone = ((xx - center_x) / max(2, box_width * 0.40)) ** 2 + ((yy - center_y) / max(2, box_height * 0.55)) ** 2 < 1
    border = (xx < width * 0.035) | (xx > width * 0.965) | (yy < height * 0.025) | (yy > height * 0.975)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness, channel_a, channel_b = cv2.split(lab)
    core_values = lab[core]
    dark_core = core & (lightness <= np.percentile(core_values[:, 0], 52))
    foreground_median = np.median(lab[dark_core], axis=0)
    foreground_mad = np.median(np.abs(lab[dark_core] - foreground_median), axis=0) + 7
    foreground_distance = np.sqrt(np.sum(((lab - foreground_median) / foreground_mad) ** 2, axis=2))

    mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    mask[border | ~subject_zone] = cv2.GC_BGD
    candidate = subject_zone & (foreground_distance < 3.4)
    if motion is not None:
        candidate |= subject_zone & (motion > 20)
    mask[candidate] = cv2.GC_PR_FGD
    if stability is not None:
        mask[(stability < 7) & ~protected_zone] = cv2.GC_BGD
    chroma = np.sqrt((channel_a - 128) ** 2 + (channel_b - 128) ** 2)
    if float(np.median(chroma[dark_core])) < 20:
        mask[(chroma > 22) & (lightness > 35) & ~core] = cv2.GC_BGD
    sure_foreground = dark_core & (foreground_distance < 1.9)
    if motion is not None:
        sure_foreground |= core & (motion > 40)
    mask[sure_foreground] = cv2.GC_FGD

    try:
        background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, background, foreground, 3, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        app.logger.exception("Foreground extraction fallback")
        alpha = np.where(subject_zone, 255, 0).astype(np.uint8)

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8), 8)
    if component_count > 1:
        scores = []
        for component in range(1, component_count):
            left, top, component_width, component_height, area = stats[component]
            component_x, component_y = centroids[component]
            distance = ((component_x - center_x) / width) ** 2 + ((component_y - center_y) / height) ** 2
            overlaps_center = left <= center_x <= left + component_width and top <= center_y <= top + component_height
            scores.append((area * (2 if overlaps_center else 1) / (1 + 8 * distance), component))
        alpha = np.where(labels == max(scores)[1], 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    foreground_pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[alpha > 127]
    if foreground_pixels.size and float(np.median(foreground_pixels)) < 85:
        gamma = 0.74 if float(np.median(foreground_pixels)) < 55 else 0.86
        enhanced = np.clip(np.power(rgb.astype(np.float32) / 255, gamma) * 255, 0, 255).astype(np.uint8)
        rgb = np.where((alpha > 0)[:, :, None], enhanced, rgb)
    return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")


def create_cutouts(paths: list[Path], job_dir: Path, count: int) -> tuple[list[Path], list[Path]]:
    cutouts, samples, records = [], [], []
    for index in range(count):
        source = paths[index % len(paths)]
        frame = source_frame(source, index // len(paths), max(1, math.ceil(count / len(paths))))
        sample = job_dir / f"sample-{index}.jpg"
        frame.convert("RGB").save(sample, quality=94)
        samples.append(sample)
        records.append((source, frame))

    detections = {}
    for source in paths:
        source_frames = [frame for record_source, frame in records if record_source == source]
        if source_frames:
            box = detect_subject(np.array(source_frames[len(source_frames) // 2].convert("RGB")))
            detections[source] = box

    contexts = {}
    for source in paths:
        source_frames = [np.array(frame.convert("RGB")) for record_source, frame in records if record_source == source]
        if len(source_frames) >= 3 and len({array.shape for array in source_frames}) == 1:
            stack = np.stack(source_frames)
            median = np.median(stack, axis=0).astype(np.uint8)
            gray_stack = np.stack([cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) for array in source_frames]).astype(np.int16)
            gray_median = np.median(gray_stack, axis=0)
            contexts[source] = (median, np.median(np.abs(gray_stack - gray_median), axis=0))

    for index, (source, frame) in enumerate(records):
        context = contexts.get(source)
        motion = None if context is None else cv2.cvtColor(cv2.absdiff(np.array(frame.convert("RGB")), context[0]), cv2.COLOR_RGB2GRAY)
        stability = None if context is None else context[1]
        result = isolate_subject(frame, motion, stability, detections.get(source))
        bounds = result.getchannel("A").getbbox()
        if bounds:
            result = result.crop(bounds)
        result.thumbnail((520, 820), Image.Resampling.LANCZOS)
        destination = job_dir / f"cutout-{index}.png"
        result.save(destination)
        cutouts.append(destination)
    return cutouts, samples


def split_title_words(_title: str) -> list[str]:
    return ["CAN", "YOU", "IMAGINE", "FLOATING", "WEIGHTLESS"]


def draw_centered_phrase(draw: ImageDraw.ImageDraw, words: list[str], y: int):
    """Draw a large phrase on at most two lines without clipping it."""
    if len(words) <= 2:
        lines = [" ".join(words)]
    else:
        split_at = math.ceil(len(words) / 2)
        lines = [" ".join(words[:split_at]), " ".join(words[split_at:])]
    text = "\n".join(lines)
    for size in range(46, 25, -2):
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, stroke_width=2, align="center")
        if box[2] - box[0] <= OUTPUT_WIDTH - 56:
            break
    width = box[2] - box[0]
    draw.multiline_text(((OUTPUT_WIDTH - width) / 2, y), text, font=font, fill="white", spacing=4, stroke_width=2, stroke_fill="black", align="center")


def make_social_storyboards(cutouts: list[Path], samples: list[Path], job_dir: Path, title: str, intro_count: int, roulette_count: int, collage_count: int) -> list[Path]:
    """Create varied stills for intro, roulette, and progressive word collage."""
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 66)
    words = split_title_words(title)
    positions = [
        (int(OUTPUT_WIDTH * 0.10), int(OUTPUT_HEIGHT * 0.13)),
        (int(OUTPUT_WIDTH * 0.51), int(OUTPUT_HEIGHT * 0.17)),
        (int(OUTPUT_WIDTH * 0.10), int(OUTPUT_HEIGHT * 0.52)),
        (int(OUTPUT_WIDTH * 0.52), int(OUTPUT_HEIGHT * 0.56)),
        (int(OUTPUT_WIDTH * 0.31), int(OUTPUT_HEIGHT * 0.32)),
        (int(OUTPUT_WIDTH * 0.04), int(OUTPUT_HEIGHT * 0.35)),
    ]
    count = intro_count + roulette_count + collage_count
    collage_start = intro_count + roulette_count
    clean_title = re.sub(r"[^A-Za-z0-9 !?-]", "", title).strip().upper()
    intro_labels = [clean_title or "LOOK", "WATCH", "IMAGINE", "FLOATING"]
    boards = []
    for index in range(count):
        canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), "black")
        raw_intro = index < intro_count and index % 2 == 0 and bool(samples)
        if raw_intro:
            full_frame = Image.open(samples[index % len(samples)]).convert("RGB")
            canvas.paste(ImageOps.fit(full_frame, canvas.size, Image.Resampling.LANCZOS))
            layers = 0
        elif index < intro_count:
            layers = 1
        elif index < collage_start:
            layers = 1 if index % 3 else 2
        else:
            layers = min(5, 2 + index - collage_start)
        for layer in range(layers):
            cutout = Image.open(cutouts[(index + layer * 3) % len(cutouts)]).convert("RGBA")
            roulette = intro_count <= index < collage_start
            if roulette and (index + layer) % 2:
                cutout = ImageOps.mirror(cutout)
            if roulette:
                angle = (-12, 7, 14, -6)[(index + layer) % 4]
                cutout = cutout.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            single_sizes = ((312, 512), (432, 680), (360, 576), (472, 720))
            max_size = single_sizes[index % len(single_sizes)] if layers == 1 else (232, 368)
            cutout.thumbnail(max_size, Image.Resampling.LANCZOS)
            if layers == 1:
                x = int((OUTPUT_WIDTH - cutout.width) / 2 + ((index % 3) - 1) * 55)
                y = int((OUTPUT_HEIGHT - cutout.height) / 2 - OUTPUT_HEIGHT * 0.035)
            else:
                x, y = positions[(index + layer) % len(positions)]
            rim_alpha = cutout.getchannel("A").filter(ImageFilter.MaxFilter(7))
            rim_alpha = rim_alpha.point(lambda value: round(value * 0.62))
            rim = Image.new("RGBA", cutout.size, (205, 225, 255, 0))
            rim.putalpha(rim_alpha)
            canvas.paste(rim, (x, y), rim)
            canvas.paste(cutout, (x, y), cutout)
        draw = ImageDraw.Draw(canvas)
        if index < intro_count:
            label = intro_labels[index % len(intro_labels)]
            box = draw.textbbox((0, 0), label, font=font_large, stroke_width=3)
            draw.text(((OUTPUT_WIDTH - (box[2] - box[0])) / 2, int(OUTPUT_HEIGHT * 0.72)), label, font=font_large, fill="white", stroke_width=3, stroke_fill="black")
        elif index >= collage_start:
            word_index = min(len(words) - 1, index - collage_start)
            word = words[word_index]
            for size in range(74, 35, -2):
                word_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
                box = draw.textbbox((0, 0), word, font=word_font, stroke_width=3)
                if box[2] - box[0] <= OUTPUT_WIDTH - 76:
                    break
            draw.text(((OUTPUT_WIDTH - (box[2] - box[0])) / 2, int(OUTPUT_HEIGHT * 0.735)), word, font=word_font, fill="white", stroke_width=3, stroke_fill="black")
        destination = job_dir / f"story-{index:02d}.jpg"
        canvas.save(destination, quality=96)
        boards.append(destination)
    return boards


def render_still_clip(image_path: Path, destination: Path, duration: float, index: int, flash: bool = False):
    zoom = 1.00 + (index % 4) * 0.035
    direction = index % 4
    x = "iw/2-(iw/zoom/2)" if direction < 2 else "iw/2-(iw/zoom/2)+12*sin(on/8)"
    y = "ih/2-(ih/zoom/2)" if direction % 2 == 0 else "ih/2-(ih/zoom/2)+10*cos(on/7)"
    filters = [f"zoompan=z='min({zoom:.3f}+on*0.0065,1.24)':x='{x}':y='{y}':d=1:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps=30"]
    if flash:
        filters.append("drawbox=color=white@0.52:t=fill:enable='lt(t,0.040)'")
    frame_count = max(1, round(duration * 30))
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", str(image_path), "-frames:v", str(frame_count), "-vf", ",".join(filters), "-an", "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-threads", "1", "-pix_fmt", "yuv420p", "-g", "15", "-keyint_min", "1", "-sc_threshold", "0", str(destination)])


def render_action_clip(source: Path, destination: Path, start: float, duration: float, index: int, flash: bool = False):
    scale = (620, 720, 820, 660)[index % 4]
    movement = 27 + (index % 4) * 10
    x = f"(in_w-out_w)/2+{movement}*sin(PI*t/{max(duration, 0.1):.3f})"
    y = f"(in_h-out_h)/2+{movement // 2}*cos(PI*t/{max(duration, 0.1):.3f})"
    speed = (1.0, 1.35, 0.82, 1.12)[index % 4]
    filters = [f"setpts=PTS/{speed:.2f}"]
    if index % 6 == 4:
        filters.append("hflip")
    angle = (0, 4.5, -5.5, 2.5)[index % 4]
    frame_count = max(1, round(duration * 30))
    filters.extend([
        f"scale={scale}:{round(scale * 16 / 9)}:force_original_aspect_ratio=increase",
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:x='{x}':y='{y}'",
        f"rotate='{angle}*PI/180*sin(PI*t/{max(duration, 0.1):.3f})':ow=iw:oh=ih:fillcolor=black,setsar=1,fps=30",
        f"zoompan=z='1+0.16*sin(PI*on/{frame_count})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps=30",
        "eq=contrast=1.12:saturation=1.24",
        "unsharp=5:5:0.5",
    ])
    if flash:
        filters.extend(["gblur=sigma=8:enable='lt(t,0.065)'", "drawbox=color=white@0.58:t=fill:enable='lt(t,0.040)'"])
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{start:.3f}", "-i", str(source), "-frames:v", str(frame_count), "-vf", ",".join(filters), "-an", "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-threads", "1", "-pix_fmt", "yuv420p", "-g", "15", "-keyint_min", "1", "-sc_threshold", "0", str(destination)])


def rhythmic_durations(start: float, end: float, count: int, onsets: list[float]) -> list[float]:
    """Create frame-exact clip lengths and nudge cuts to nearby audio attacks."""
    fps, minimum_frames = 30, 4
    start_frame, end_frame = round(start * fps), round(end * fps)
    onset_frames = [round(value * fps) for value in onsets if start < value < end]
    boundaries = [start_frame]
    for index in range(1, count):
        target = round(start_frame + (end_frame - start_frame) * index / count)
        earliest = boundaries[-1] + minimum_frames
        latest = end_frame - minimum_frames * (count - index)
        nearby = [frame for frame in onset_frames if abs(frame - target) <= 5 and earliest <= frame <= latest]
        boundaries.append(min(nearby, key=lambda frame: abs(frame - target)) if nearby else max(earliest, min(latest, target)))
    boundaries.append(end_frame)
    return [(boundaries[index + 1] - boundaries[index]) / fps for index in range(count)]


def snap_to_onset(target: float, onsets: list[float], radius: float = 0.24) -> float:
    nearby = [onset for onset in onsets if abs(onset - target) <= radius]
    return min(nearby, key=lambda onset: abs(onset - target)) if nearby else target


def render_floating_video(cutouts: list[Path], samples: list[Path], paths: list[Path], audio_path: Path, job_dir: Path, destination: Path, duration: float, rhythm: dict, title: str) -> int:
    """Reference-like structure: intro → cutout roulette → word collage → action climax."""
    edit_bpm = rhythm["edit_bpm"]
    beat = 60 / edit_bpm
    has_video = any(path.suffix[1:] in ALLOWED_VIDEO for path in paths)
    black_share = 0.48 if has_video else 1.0
    black_duration = snap_to_onset(duration * black_share, rhythm["onsets"])
    climax_duration = max(0, duration - black_duration)

    intro_end = snap_to_onset(black_duration * 0.27, rhythm["onsets"])
    roulette_end = snap_to_onset(black_duration * 0.73, rhythm["onsets"])
    intro_count = 4 if duration >= 8 else 2
    collage_count = 5
    roulette_count = max(4, min(22, round((roulette_end - intro_end) / max(beat * 0.75, 0.20))))
    storyboards = make_social_storyboards(cutouts, samples, job_dir, title, intro_count, roulette_count, collage_count)
    clip_durations = (
        rhythmic_durations(0, intro_end, intro_count, rhythm["onsets"])
        + rhythmic_durations(intro_end, roulette_end, roulette_count, rhythm["onsets"])
        + rhythmic_durations(roulette_end, black_duration, collage_count, rhythm["onsets"])
    )

    clips = []
    for index, board in enumerate(storyboards):
        clip = job_dir / f"clip-{len(clips):03d}.mp4"
        section_change = index in {intro_count, intro_count + roulette_count}
        roulette_accent = intro_count <= index < intro_count + roulette_count and (index - intro_count) % 4 == 0
        render_still_clip(board, clip, clip_durations[index], index, flash=section_change or roulette_accent)
        clips.append(clip)

    if has_video and climax_duration > 0:
        action_count = max(5, min(30, round(climax_duration / max(beat * 0.82, 0.20))))
        action_durations = rhythmic_durations(black_duration, duration, action_count, rhythm["onsets"])
        video_paths = [path for path in paths if path.suffix[1:] in ALLOWED_VIDEO]
        for index, action_duration in enumerate(action_durations):
            source = video_paths[index % len(video_paths)]
            source_length = media_duration(source)
            start = max(0, (source_length - action_duration) * ((index * 0.61803398875) % 1))
            clip = job_dir / f"clip-{len(clips):03d}.mp4"
            render_action_clip(source, clip, start, action_duration, index, flash=index == 0 or index % 4 == 0)
            clips.append(clip)

    concat_file = job_dir / "clips.txt"
    concat_file.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-i", str(audio_path), "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
    return len(clips)


def keep_only_output(job_dir: Path, destination: Path | None = None):
    """Keep Render's small ephemeral disk from filling with intermediate clips."""
    for path in job_dir.iterdir():
        if destination is not None and path == destination:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/api/health")
def health():
    return jsonify(status="ok", ffmpeg=shutil.which("ffmpeg") is not None, output=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}", profile="floating-memory-safe")


@app.post("/api/analyze-audio")
def analyze_audio():
    audio = request.files.get("audio")
    style = request.form.get("style", "cinematic")
    requested_seconds = min(60, max(5, int(request.form.get("duration", "15"))))
    if not audio or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Carica una canzone in formato MP3, WAV, M4A, AAC o OGG.")
    with tempfile.TemporaryDirectory(prefix="petcut-analysis-") as temp:
        path = Path(temp) / f"audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
        audio.save(path)
        rhythm = analyze_rhythm(path)
        usable_duration = min(float(requested_seconds), media_duration(path))
    contents = recommended_content(usable_duration, style)
    cuts = recommended_scenes(usable_duration, rhythm["edit_bpm"], style)
    return jsonify(bpm=rhythm["bpm"], edit_bpm=rhythm["edit_bpm"], recommended_content=contents, visual_cuts=cuts, message=(
        f"Ritmo musicale: circa {rhythm['bpm']} BPM; montaggio a {rhythm['edit_bpm']} BPM. "
        f"Per {usable_duration:.0f} secondi sono ideali {contents} foto o clip distinti. "
        f"Se ne carichi uno solo, PetCut creerà comunque circa {cuts} cambi ritmici riutilizzando inquadrature diverse."
    ))


def render_job(job_id, paths, audio_path, style, title, requested_seconds, job_dir):
    try:
        duration = min(float(requested_seconds), media_duration(audio_path))
        rhythm = analyze_rhythm(audio_path)
        edit_bpm = rhythm["edit_bpm"]
        if style == "floating":
            cutout_count = max(5, min(7, recommended_scenes(duration, edit_bpm, style)))
            try:
                cutouts, samples = create_cutouts(paths, job_dir, cutout_count)
            finally:
                release_subject_detector()
            destination = job_dir / "petcut-social-edit.mp4"
            scene_count = render_floating_video(cutouts, samples, paths, audio_path, job_dir, destination, duration, rhythm, title)
            JOBS[job_id].update(status="complete", output=str(destination), bpm=rhythm["bpm"], edit_bpm=edit_bpm, scenes=scene_count)
            keep_only_output(job_dir, destination)
            return
        scenes = recommended_scenes(duration, edit_bpm, style)
        scene_duration = duration / scenes
        command = ["ffmpeg", "-y"]
        for path in paths:
            command.extend((["-loop", "1", "-framerate", "30", "-i", str(path)] if path.suffix[1:] in ALLOWED_IMAGE else ["-stream_loop", "-1", "-i", str(path)]))
        command.extend(["-i", str(audio_path)])
        labels, filters = [], []
        for index in range(scenes):
            label, input_index = f"s{index}", index % len(paths)
            source_length = 1 if paths[input_index].suffix[1:] in ALLOWED_IMAGE else media_duration(paths[input_index])
            start = 0 if source_length <= scene_duration else (source_length - scene_duration) * (index // len(paths)) / max(1, (scenes - 1) // len(paths))
            filters.append(f"[{input_index}:v]trim=start={start:.3f}:duration={scene_duration:.3f},setpts=PTS-STARTPTS,{scene_filter(index, style, edit_bpm, scene_duration)}[{label}]")
            labels.append(f"[{label}]")
        filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={scenes}:v=1:a=0[m];[m]{final_filter(style, title)}[outv]"
        destination = job_dir / "petcut-social-edit.mp4"
        command.extend(["-t", f"{duration:.2f}", "-filter_complex", filter_complex, "-map", "[outv]", "-map", f"{len(paths)}:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
        run(command)
        JOBS[job_id].update(status="complete", output=str(destination), bpm=rhythm["bpm"], edit_bpm=edit_bpm, scenes=scenes)
        keep_only_output(job_dir, destination)
    except Exception as error:
        app.logger.exception("Render failed")
        if isinstance(error, subprocess.CalledProcessError):
            app.logger.error(error.stderr)
        JOBS[job_id].update(status="failed", error="Generazione non riuscita. Prova un video più breve o riprova tra poco.")
        keep_only_output(job_dir)


@app.post("/api/render")
def render_video():
    media, audio = [file for file in request.files.getlist("media") if file and file.filename], request.files.get("audio")
    style, title = request.form.get("style", "cinematic"), request.form.get("title", "").strip()
    requested_seconds = min(60, max(5, int(request.form.get("duration", "15"))))
    if not media or not audio:
        abort(400, "Carica almeno una foto o un video e una canzone.")
    if style not in {"cinematic", "beat", "collage", "floating"} or any(not extension_allowed(file.filename, ALLOWED_VIDEO | ALLOWED_IMAGE) for file in media) or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Formato o preset non supportato.")
    job_id, job_dir = uuid.uuid4().hex, OUTPUT_DIR / "jobs" / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / f"audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
    audio.save(audio_path)
    paths = []
    for index, file in enumerate(media):
        path = job_dir / f"media-{index}.{secure_filename(file.filename).rsplit('.', 1)[1].lower()}"
        file.save(path)
        paths.append(path)
    JOBS[job_id] = {"status": "processing"}
    RENDER_QUEUE.submit(render_job, job_id, paths, audio_path, style, title, requested_seconds, job_dir)
    return jsonify(job_id=job_id), 202


@app.get("/api/render/<job_id>")
def render_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404, "Job non trovato. Riprova a generare il video.")
    return jsonify({key: value for key, value in job.items() if key != "output"})


@app.get("/api/render/<job_id>/download")
def download_render(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "complete":
        abort(404, "Il video non è ancora pronto.")
    return send_file(job["output"], as_attachment=True, download_name="petcut-social-edit.mp4", mimetype="video/mp4")


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"File troppo grande: limite {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
