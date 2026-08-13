import math
import os
import re
import gc
import hashlib
import json
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
import onnxruntime as ort
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from audio_analysis import analyze_audio as analyze_audio_structure
from audio_analysis import recommendation as audio_recommendation
from render_engine import render_preset

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "exports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO = {"mp4", "mov", "m4v", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}
ALLOWED_AUDIO = {"mp3", "wav", "m4a", "aac", "ogg"}
REFERENCE_STYLE = "reference_edit"
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
FOREGROUND_NET = None
FOREGROUND_NET_FAILED = False
FOREGROUND_NET_LOCK = threading.Lock()
INSTANCE_NET = None
INSTANCE_NET_FAILED = False
INSTANCE_NET_LOCK = threading.Lock()
MODEL_DIR = Path(os.environ.get("PETCUT_MODEL_DIR", BASE_DIR / "data" / "models"))
INSTANCE_MODEL_SHA256 = "c00375e81c9b2793d12f6473c6fd477ba5cce20710c2c5f1ae2118f4af345112"
YOLO_CFG = MODEL_DIR / "yolov4-tiny.cfg"
YOLO_WEIGHTS = MODEL_DIR / "yolov4-tiny.weights"
YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
SUBJECT_CLASSES = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
U2NETP_MODEL = MODEL_DIR / "u2netp.onnx"
U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
U2NETP_MD5 = "8e83ca70e441ab06c318d82300c84806"
INSTANCE_MODEL = MODEL_DIR / "yolov8n-seg.onnx"
# COCO foreground categories useful for people and social animal edits.
INSTANCE_CLASSES = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def extension_allowed(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def requested_duration() -> int:
    """Read a bounded duration without turning malformed form data into a 500."""
    try:
        value = int(request.form.get("duration", "24"))
    except (TypeError, ValueError):
        abort(400, "Durata non valida.")
    return min(60, max(5, value))


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
        frame = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
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


def download_checked_model(path: Path, url: str, expected_md5: str, minimum_size: int):
    """Download a small inference model atomically and verify its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if path.stat().st_size >= minimum_size and digest == expected_md5:
            return
        path.unlink(missing_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.md5(temporary.read_bytes()).hexdigest()
    if temporary.stat().st_size < minimum_size or digest != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Modello di segmentazione incompleto")
    temporary.replace(path)


def foreground_detector():
    """Load the lightweight U2NetP saliency network through OpenCV DNN."""
    global FOREGROUND_NET, FOREGROUND_NET_FAILED
    if FOREGROUND_NET is not None:
        return FOREGROUND_NET
    if FOREGROUND_NET_FAILED:
        return None
    with FOREGROUND_NET_LOCK:
        if FOREGROUND_NET is not None:
            return FOREGROUND_NET
        if FOREGROUND_NET_FAILED:
            return None
        try:
            download_checked_model(U2NETP_MODEL, U2NETP_URL, U2NETP_MD5, 4_000_000)
            FOREGROUND_NET = cv2.dnn.readNetFromONNX(str(U2NETP_MODEL))
        except Exception:
            FOREGROUND_NET_FAILED = True
            app.logger.exception("Foreground model unavailable; using GrabCut fallback")
    return FOREGROUND_NET


def neural_saliency(rgb: np.ndarray) -> np.ndarray | None:
    """Return a soft foreground probability map at the source resolution."""
    network = foreground_detector()
    if network is None:
        return None
    try:
        resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        resized = (resized - np.array((0.485, 0.456, 0.406), np.float32)) / np.array((0.229, 0.224, 0.225), np.float32)
        blob = np.transpose(resized, (2, 0, 1))[None]
        with FOREGROUND_NET_LOCK:
            network.setInput(blob)
            output = network.forward(network.getUnconnectedOutLayersNames())[0][0, 0]
        low, high = float(np.min(output)), float(np.max(output))
        if high - low < 1e-6:
            return None
        output = np.clip((output - low) / (high - low), 0, 1)
        return cv2.resize(output, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    except Exception:
        app.logger.exception("Foreground inference failed; using GrabCut fallback")
        return None


def instance_segmenter():
    """Load the compact animal/person instance segmenter with bounded threads."""
    global INSTANCE_NET, INSTANCE_NET_FAILED
    if INSTANCE_NET is not None:
        return INSTANCE_NET
    if INSTANCE_NET_FAILED or not INSTANCE_MODEL.exists():
        return None
    with INSTANCE_NET_LOCK:
        if INSTANCE_NET is not None:
            return INSTANCE_NET
        if INSTANCE_NET_FAILED:
            return None
        try:
            digest_path = INSTANCE_MODEL.with_suffix(".sha256-ok")
            if not digest_path.exists():
                digest = hashlib.sha256(INSTANCE_MODEL.read_bytes()).hexdigest()
                if digest != INSTANCE_MODEL_SHA256:
                    raise RuntimeError("Modello ONNX di segmentazione non valido")
                try:
                    digest_path.write_text(digest, encoding="ascii")
                except OSError:
                    pass
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            INSTANCE_NET = ort.InferenceSession(
                str(INSTANCE_MODEL),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            INSTANCE_NET_FAILED = True
            app.logger.exception("Instance segmenter unavailable; using saliency fallback")
    return INSTANCE_NET


def instance_subject_mask(rgb: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Return the single main person/animal instance and its tight box.

    Unioning every detection looked disastrous when a dog sat beside a
    person: trousers, furniture and the animal became one cutout. Animal
    instances are preferred when present; otherwise the most central,
    confident person is used.
    """
    session = instance_segmenter()
    if session is None:
        return None, None
    try:
        height, width = rgb.shape[:2]
        size = 640
        scale = min(size / width, size / height)
        scaled_width, scaled_height = round(width * scale), round(height * scale)
        resized = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        offset_x, offset_y = (size - scaled_width) // 2, (size - scaled_height) // 2
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width] = resized
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        with INSTANCE_NET_LOCK:
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        detection = next(value for value in outputs if value.ndim == 3 and value.shape[1] > 80)
        prototype = next(value for value in outputs if value.ndim == 4 and value.shape[1] == 32)[0]
        predictions = detection[0].T
        class_scores = predictions[:, 4:84]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        candidates = np.flatnonzero((confidences >= 0.12) & np.isin(class_ids, list(INSTANCE_CLASSES)))
        if not len(candidates):
            return None, None

        boxes = []
        for candidate in candidates:
            center_x, center_y, box_width, box_height = predictions[candidate, :4]
            boxes.append([float(center_x - box_width / 2), float(center_y - box_height / 2), float(box_width), float(box_height)])
        selected = cv2.dnn.NMSBoxes(boxes, confidences[candidates].tolist(), 0.12, 0.55)
        if len(selected) == 0:
            return None, None

        selected_indices = np.asarray(selected).reshape(-1).astype(int).tolist()
        animal_indices = [
            selected_index
            for selected_index in selected_indices
            if int(class_ids[candidates[selected_index]]) != 0
        ]
        primary_pool = animal_indices or selected_indices

        def primary_score(selected_index: int) -> float:
            candidate = int(candidates[selected_index])
            box_x, box_y, box_width, box_height = boxes[selected_index]
            center_x = box_x + box_width / 2
            center_y = box_y + box_height / 2
            centrality = max(
                0.0,
                1.0
                - abs(center_x - size / 2) / (size / 2)
                - abs(center_y - size / 2) / (size / 2),
            )
            area = min(1.0, max(0.0, box_width * box_height / (size * size * 0.45)))
            return float(confidences[candidate]) * 0.64 + centrality * 0.22 + area * 0.14

        selected_index = max(primary_pool, key=primary_score)
        candidate = int(candidates[selected_index])
        proto_height, proto_width = prototype.shape[1:]
        xx = np.arange(proto_width)[None, :]
        yy = np.arange(proto_height)[:, None]
        coefficients = predictions[candidate, 84:]
        logits = np.clip(coefficients @ prototype.reshape(32, -1), -30, 30)
        probability = (1.0 / (1.0 + np.exp(-logits))).reshape(proto_height, proto_width)
        box_x, box_y, box_width, box_height = boxes[selected_index]
        inside = (
            (xx >= box_x * proto_width / size)
            & (xx < (box_x + box_width) * proto_width / size)
            & (yy >= box_y * proto_height / size)
            & (yy < (box_y + box_height) * proto_height / size)
        )
        probability *= inside
        probability = cv2.resize(probability, (size, size), interpolation=cv2.INTER_LINEAR)
        probability = probability[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width]
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LANCZOS4)
        binary = (probability > 0.34).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if component_count > 1:
            keep = np.zeros_like(binary)
            for component in range(1, component_count):
                if stats[component, cv2.CC_STAT_AREA] >= max(80, width * height * 0.002):
                    keep[labels == component] = 1
            binary = keep
        probability *= cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
        points = cv2.findNonZero((binary * 255).astype(np.uint8))
        if points is None:
            return None, None
        box = cv2.boundingRect(points)
        return probability.astype(np.float32), box
    except Exception:
        app.logger.exception("Instance segmentation failed; using saliency fallback")
        return None, None


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
    global SUBJECT_NET, FOREGROUND_NET, INSTANCE_NET, INSTANCE_NET_FAILED
    with SUBJECT_NET_LOCK:
        SUBJECT_NET = None
    with FOREGROUND_NET_LOCK:
        FOREGROUND_NET = None
    with INSTANCE_NET_LOCK:
        INSTANCE_NET = None
        # A miss disables the heavier model only for the current job. Allow
        # the next upload to try it again after all native buffers are freed.
        INSTANCE_NET_FAILED = False
    gc.collect()


def release_instance_after_miss():
    """Free ONNX buffers before loading the lightweight fallback networks."""
    global INSTANCE_NET, INSTANCE_NET_FAILED
    with INSTANCE_NET_LOCK:
        INSTANCE_NET = None
        INSTANCE_NET_FAILED = True
    gc.collect()


def isolate_subject(frame: Image.Image, motion: np.ndarray | None = None, stability: np.ndarray | None = None, subject_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Create a clean alpha matte using saliency, detection and GrabCut together."""
    rgb = np.array(frame.convert("RGB"))
    height, width = rgb.shape[:2]
    instance_probability, instance_box = instance_subject_mask(rgb)
    if instance_probability is None:
        # Avoid retaining YOLOv8's native inference workspace while U2NetP
        # and GrabCut are active; this is important on 512 MB Render workers.
        release_instance_after_miss()
        # Do not stack two more neural networks after an instance miss. The
        # conservative centre box below feeds the existing colour-aware
        # GrabCut fallback and stays well within Render Free's memory limit.
        fallback_box = (
            int(width * 0.12),
            int(height * 0.08),
            int(width * 0.76),
            int(height * 0.86),
        )
        x, y, box_width, box_height = subject_box or fallback_box
    else:
        x, y, box_width, box_height = instance_box or subject_box or detect_subject(rgb)
    x, y = max(0, x), max(0, y)
    box_width = max(2, min(width - x, box_width))
    box_height = max(2, min(height - y, box_height))

    yy, xx = np.mgrid[:height, :width]
    center_x, center_y = x + box_width * 0.5, y + box_height * 0.48
    expanded_left = max(0, int(x - box_width * 0.08))
    expanded_top = max(0, int(y - box_height * 0.08))
    expanded_right = min(width, int(x + box_width * 1.08))
    expanded_bottom = min(height, int(y + box_height * 1.08))
    subject_zone = (xx >= expanded_left) & (xx < expanded_right) & (yy >= expanded_top) & (yy < expanded_bottom)
    core = ((xx - center_x) / max(2, box_width * 0.22)) ** 2 + ((yy - (y + box_height * 0.40)) / max(2, box_height * 0.34)) ** 2 < 1
    face_zone = ((xx - center_x) / max(2, box_width * 0.23)) ** 2 + ((yy - (y + box_height * 0.30)) / max(2, box_height * 0.23)) ** 2 < 1
    protected_zone = ((xx - center_x) / max(2, box_width * 0.38)) ** 2 + ((yy - center_y) / max(2, box_height * 0.54)) ** 2 < 1
    border = (xx < width * 0.035) | (xx > width * 0.965) | (yy < height * 0.025) | (yy > height * 0.975)
    saliency = instance_probability

    if instance_probability is not None:
        # Instance masks are much cleaner around rugs, furniture and floors
        # than a generic salient-object ellipse. Preserve a soft 2-pixel edge
        # while keeping the learned mask itself authoritative.
        # Keep the learned matte inside the detected instance.  The former
        # low threshold retained a coloured fringe of rugs, walls and beds;
        # that fringe became especially obvious against the pure-black
        # roulette background.
        alpha = np.clip((instance_probability - 0.30) / 0.46, 0, 1)
        alpha = (alpha * 255).astype(np.uint8)
        lab_instance = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        instance_lightness, instance_a, instance_b = cv2.split(lab_instance)
        instance_chroma = np.sqrt((instance_a - 128) ** 2 + (instance_b - 128) ** 2)
        confident = alpha > 210
        if np.any(confident) and float(np.median(instance_chroma[confident])) < 16 and float(np.median(instance_lightness[confident])) < 72:
            coloured_background = (instance_chroma > 19) & (instance_lightness > 28) & ~face_zone
            alpha[coloured_background] = 0
        # Preserve fur detail but avoid the light fringe produced by dilating
        # the learned mask into the original background.
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        foreground_pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[alpha > 127]
        if foreground_pixels.size and float(np.median(foreground_pixels)) < 95:
            # Lift luminance in Lab space so black fur stays coloured and its
            # texture remains visible against the roulette's black canvas.
            # A linear lift barely changed near-black pixels; the adaptive
            # curve targets those shadows without clipping eyes/highlights.
            median_luma = float(np.median(foreground_pixels))
            gamma = 0.48 if median_luma < 45 else 0.68
            lab_enhanced = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            light = lab_enhanced[:, :, 0].astype(np.float32) / 255.0
            lab_enhanced[:, :, 0] = np.clip(np.power(light, gamma) * 198 + 50, 0, 255).astype(np.uint8)
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.75)
            enhanced = cv2.addWeighted(enhanced, 1.38, blurred, -0.38, 0)
            rgb = np.where((alpha > 0)[:, :, None], enhanced, rgb)
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness, channel_a, channel_b = cv2.split(lab)
    seed_zone = core if saliency is None else core & (saliency > 0.52)
    if np.count_nonzero(seed_zone) < 40:
        seed_zone = core
    core_values = lab[seed_zone]
    core_limit = np.percentile(core_values[:, 0], 68)
    foreground_seeds = seed_zone & (lightness <= core_limit)
    if np.count_nonzero(foreground_seeds) < 40:
        foreground_seeds = seed_zone
    foreground_median = np.median(lab[foreground_seeds], axis=0)
    foreground_mad = np.median(np.abs(lab[foreground_seeds] - foreground_median), axis=0) + 7
    foreground_distance = np.sqrt(np.sum(((lab - foreground_median) / foreground_mad) ** 2, axis=2))

    mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    mask[border | ~subject_zone] = cv2.GC_BGD
    if saliency is not None:
        mask[(saliency < 0.03) | ~subject_zone] = cv2.GC_BGD
        candidate = subject_zone & (saliency > 0.18)
    else:
        candidate = subject_zone & (foreground_distance < 3.8)
    if motion is not None:
        candidate |= subject_zone & protected_zone & (motion > 18)
    mask[candidate] = cv2.GC_PR_FGD
    if stability is not None:
        mask[(stability < 7) & ~protected_zone] = cv2.GC_BGD
    chroma = np.sqrt((channel_a - 128) ** 2 + (channel_b - 128) ** 2)
    sure_foreground = foreground_seeds & (foreground_distance < 2.2)
    if saliency is not None:
        sure_foreground |= subject_zone & (saliency > 0.84) & (foreground_distance < 2.9)
    if motion is not None:
        sure_foreground |= core & (motion > 38)
    mask[sure_foreground] = cv2.GC_FGD

    # A neutral, very dark subject is common in the reference animal edits.
    # In that case, bright/chromatic stable areas are almost certainly rugs,
    # furniture or floor. Apply this only for that conservative case so people
    # and coloured animals keep their clothes/fur.
    if float(foreground_median[0]) < 62 and float(np.median(chroma[foreground_seeds])) < 18:
        coloured_background = (chroma > 16) & (lightness > 28) & ~face_zone
        lower_light_background = (yy > y + box_height * 0.62) & (lightness > 92) & (chroma < 13)
        mask[coloured_background | lower_light_background] = cv2.GC_BGD

    try:
        background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, background, foreground, 4, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        app.logger.exception("Foreground extraction fallback")
        if saliency is not None:
            alpha = np.clip((saliency - 0.10) / 0.62, 0, 1)
            alpha = (alpha * 255).astype(np.uint8)
        else:
            alpha = np.where(subject_zone, 255, 0).astype(np.uint8)

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8), 8)
    if component_count > 1:
        keep = np.zeros_like(alpha)
        for component in range(1, component_count):
            left, top, component_width, component_height, area = stats[component]
            component_x, component_y = centroids[component]
            distance = ((component_x - center_x) / width) ** 2 + ((component_y - center_y) / height) ** 2
            overlaps_center = left <= center_x <= left + component_width and top <= center_y <= top + component_height
            overlaps_subject = np.count_nonzero((labels == component) & protected_zone)
            if area >= 80 and (overlaps_center or overlaps_subject >= max(20, int(area * 0.02))) and distance < 0.28:
                keep[labels == component] = 255
        if np.any(keep):
            alpha = keep
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    foreground_pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[alpha > 127]
    if foreground_pixels.size and float(np.median(foreground_pixels)) < 95:
        median_luma = float(np.median(foreground_pixels))
        gamma = 0.48 if median_luma < 45 else 0.68
        lab_enhanced = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        light = lab_enhanced[:, :, 0].astype(np.float32) / 255.0
        lab_enhanced[:, :, 0] = np.clip(np.power(light, gamma) * 198 + 50, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.75)
        enhanced = cv2.addWeighted(enhanced, 1.38, blurred, -0.38, 0)
        rgb = np.where((alpha > 0)[:, :, None], enhanced, rgb)
    return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")


def cutout_quality(subject: Image.Image) -> float:
    """Score a matte by shape and retained detail, never by identity.

    This is deliberately relative: a valid person can have a less convex
    silhouette than a dog, but a rectangle of bed/wall is still penalised.
    `create_cutouts` keeps the best candidates from the current upload rather
    than enforcing one universal threshold across pets and people.
    """
    alpha = np.asarray(subject.convert("RGBA").getchannel("A"), dtype=np.uint8)
    points = cv2.findNonZero((alpha > 127).astype(np.uint8))
    if points is None:
        return 0.0
    left, top, width, height = cv2.boundingRect(points)
    if width < 4 or height < 4:
        return 0.0
    cropped = (alpha[top : top + height, left : left + width] > 127).astype(np.uint8)
    area = float(np.count_nonzero(cropped))
    fill = area / max(1.0, float(width * height))
    perimeter = max(1.0, float(2 * width + 2 * height - 4))
    edge = (
        np.count_nonzero(cropped[0])
        + np.count_nonzero(cropped[-1])
        + np.count_nonzero(cropped[1:-1, 0])
        + np.count_nonzero(cropped[1:-1, -1])
    ) / perimeter
    contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    main = max(contours, key=cv2.contourArea)
    hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(main))))
    solidity = min(1.0, float(cv2.contourArea(main)) / hull_area)
    significant = sum(cv2.contourArea(contour) >= area * 0.035 for contour in contours)

    # Neural mattes occasionally retain the outer silhouette while punching
    # dozens of transparent islands through the animal (eyes, muzzle, hands
    # and furniture in the same mask).  Shape/solidity alone scores those
    # masks surprisingly well.  Legitimate gaps between legs live mostly in
    # the lower third and there are normally only one or two; several sizeable
    # holes through the upper two-thirds are therefore a reliable corruption
    # signal without penalising a dark or low-contrast subject.
    inverse = (1 - cropped).astype(np.uint8)
    padded = np.pad(inverse, 1, constant_values=1)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 2)
    enclosed_holes = (flooded[1:-1, 1:-1] == 1).astype(np.uint8)
    hole_ratio = float(np.count_nonzero(enclosed_holes)) / max(1.0, area)
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(enclosed_holes, 8)
    minimum_hole = max(20.0, area * 0.00025)
    significant_holes = sum(
        hole_stats[index, cv2.CC_STAT_AREA] >= minimum_hole
        for index in range(1, hole_count)
    )
    if hole_ratio >= 0.06 and significant_holes >= 3:
        return 0.0

    score = 1.0
    # The matte is deliberately cropped to its alpha bounds, so a few paws,
    # ears or the top of a portrait touching that new edge are normal.  Only
    # broad edge contact is evidence that a rectangular piece of background
    # survived segmentation.
    score -= min(0.65, max(0.0, edge - 0.28) * 2.3)
    score -= max(0.0, 0.68 - solidity) * 2.8
    score -= max(0.0, fill - 0.85) * 2.8
    score -= max(0, significant - 1) * 0.10
    rgba = np.asarray(subject.convert("RGBA"), dtype=np.uint8)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    visible = rgba[:, :, 3] > 127
    contrast = float(np.std(gray[visible])) if np.any(visible) else 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(laplacian[visible])) if np.any(visible) else 0.0
    appearance = 0.60 * min(1.0, contrast / 35.0) + 0.40 * min(1.0, sharpness / 180.0)
    score *= 0.35 + 0.65 * appearance
    return float(max(0.0, min(1.0, score)))


def create_cutouts(paths: list[Path], job_dir: Path, count: int) -> tuple[list[Path], list[Path]]:
    samples, records = [], []
    for index in range(count):
        source = paths[index % len(paths)]
        frame = source_frame(source, index // len(paths), max(1, math.ceil(count / len(paths))))
        sample = job_dir / f"sample-{index:02d}-source-{index % len(paths):02d}.jpg"
        frame.convert("RGB").save(sample, quality=94)
        samples.append(sample)
        records.append((source, frame))

    candidates: list[tuple[float, Path, Path]] = []
    for index, (source, frame) in enumerate(records):
        # Frames sampled seconds apart often belong to different shots.  A
        # temporal median across those shots marks the whole changed room as
        # foreground, so segmentation must be evaluated independently.
        result = isolate_subject(frame)
        quality = cutout_quality(result)
        bounds = result.getchannel("A").getbbox()
        if bounds:
            result = result.crop(bounds)
        result.thumbnail((520, 820), Image.Resampling.LANCZOS)
        destination = job_dir / f"cutout-{index}.png"
        result.save(destination)
        candidates.append((quality, destination, samples[index]))

    # Reject obvious rectangular/background mattes while remaining useful
    # when the upload contains only people or a single difficult frame.  The
    # renderer can safely reuse a clean pose; one broken matte is far more
    # noticeable than a repeated clean one.
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    best = ranked[0][0] if ranked else 0.0
    selected = [item for item in candidates if item[0] >= max(0.30, best - 0.12)]
    minimum = min(len(ranked), 1)
    if len(selected) < minimum:
        selected_paths = {path for _, path, _ in selected}
        selected.extend(item for item in ranked if item[1] not in selected_paths and len(selected) < minimum)
    if not selected:
        selected = ranked[:1]
    cutouts = [path for _, path, _ in selected]
    selected_samples = [sample for _, _, sample in selected]
    return cutouts, selected_samples


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
        if (destination is not None and path == destination) or path.name == "job.json":
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
    return jsonify(
        status="ok",
        ffmpeg=shutil.which("ffmpeg") is not None,
        output=f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        fps=30,
        mode=REFERENCE_STYLE,
        profile="reference-match-v5",
    )


@app.post("/api/analyze-audio")
def analyze_audio():
    audio = request.files.get("audio")
    requested_seconds = requested_duration()
    if not audio or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Carica una canzone in formato MP3, WAV, M4A, AAC o OGG.")
    with tempfile.TemporaryDirectory(prefix="petcut-analysis-") as temp:
        path = Path(temp) / f"audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
        audio.save(path)
        rhythm = analyze_audio_structure(path, requested_seconds)
    plan = audio_recommendation(REFERENCE_STYLE, rhythm["duration"], rhythm)
    return jsonify(
        bpm=rhythm["bpm"],
        edit_bpm=rhythm["edit_bpm"],
        duration=rhythm["duration"],
        drop_time=rhythm["drop_time"],
        sections=rhythm["sections"],
        phases=plan["phases"],
        recommended_content=plan["ideal_media"],
        minimum_content=plan["min_media"],
        visual_cuts=plan["visual_cuts"],
        message=plan["note"],
    )


def render_job(job_id, paths, audio_path, title, requested_seconds, job_dir):
    try:
        JOBS[job_id].update(progress=3, stage=0, phase="Analisi del ritmo e delle sezioni musicali")
        rhythm = analyze_audio_structure(audio_path, float(requested_seconds))
        duration = min(float(requested_seconds), float(rhythm["duration"]))
        JOBS[job_id].update(progress=10, stage=1, phase="Selezione delle inquadrature migliori")

        # The only edit uses layered subjects throughout its first three acts.
        # Twelve temporal samples preserve pose variety even with one source.
        has_video = any(Path(path).suffix.lower().lstrip(".") in ALLOWED_VIDEO for path in paths)
        # A still image yields the same matte on every pass; reuse it through
        # transforms in the renderer instead of running segmentation eight
        # times. Videos still supply twelve genuinely different poses.
        cutout_count = 12 if has_video else max(1, min(12, len(paths)))
        try:
            cutouts, samples = create_cutouts(paths, job_dir, cutout_count)
        finally:
            release_subject_detector()
        JOBS[job_id].update(progress=26, stage=2, phase="Soggetti scontornati e livelli pronti")

        destination = job_dir / "petcut-social-edit.mp4"

        def update_progress(phase, percent):
            value = max(int(JOBS[job_id].get("progress", 0)), int(percent))
            stage = 0 if value < 10 else 1 if value < 26 else 2 if value < 94 else 3
            JOBS[job_id].update(progress=value, stage=stage, phase=phase.capitalize())

        metadata = render_preset(
            REFERENCE_STYLE,
            paths,
            audio_path,
            job_dir,
            destination,
            duration,
            rhythm,
            title,
            cutouts=cutouts,
            samples=samples,
            progress=update_progress,
        )
        JOBS[job_id].update(
            status="complete",
            output=str(destination),
            progress=100,
            stage=4,
            phase="Video completato",
            bpm=rhythm["bpm"],
            edit_bpm=rhythm["edit_bpm"],
            drop_time=metadata.get("drop_time", rhythm["drop_time"]),
            mode=REFERENCE_STYLE,
            scenes=metadata["scenes"],
        )
        (job_dir / "job.json").write_text(
            json.dumps({"job_id": job_id, **{key: value for key, value in JOBS[job_id].items() if key != "output"}}),
            encoding="utf-8",
        )
        keep_only_output(job_dir, destination)
    except Exception as error:
        app.logger.exception("Render failed")
        if isinstance(error, subprocess.CalledProcessError):
            app.logger.error(error.stderr)
        JOBS[job_id].update(status="failed", phase="Montaggio interrotto", error="Generazione non riuscita. Prova con file più brevi o riprova tra poco.")
        keep_only_output(job_dir)


@app.post("/api/render")
def render_video():
    media, audio = [file for file in request.files.getlist("media") if file and file.filename], request.files.get("audio")
    title = request.form.get("title", "").strip()[:64]
    requested_seconds = requested_duration()
    if not media or not audio:
        abort(400, "Carica almeno una foto o un video e una canzone.")
    if any(not extension_allowed(file.filename, ALLOWED_VIDEO | ALLOWED_IMAGE) for file in media) or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Formato non supportato.")
    if len(media) > 30:
        abort(400, "Puoi caricare al massimo 30 foto o video per montaggio.")
    job_id, job_dir = uuid.uuid4().hex, OUTPUT_DIR / "jobs" / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / f"audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
    audio.save(audio_path)
    paths = []
    for index, file in enumerate(media):
        path = job_dir / f"media-{index}.{secure_filename(file.filename).rsplit('.', 1)[1].lower()}"
        file.save(path)
        paths.append(path)
    JOBS[job_id] = {"status": "processing", "progress": 1, "stage": 0, "phase": "File ricevuti"}
    RENDER_QUEUE.submit(render_job, job_id, paths, audio_path, title, requested_seconds, job_dir)
    return jsonify(job_id=job_id), 202


@app.get("/api/render/<job_id>")
def render_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        job_root = OUTPUT_DIR / "jobs"
        for metadata_path in job_root.glob("*/job.json"):
            try:
                candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if candidate.get("job_id") == job_id:
                output = metadata_path.parent / "petcut-social-edit.mp4"
                if output.exists():
                    candidate["output"] = str(output)
                    JOBS[job_id] = candidate
                    job = candidate
                break
    if not job:
        abort(404, "Job non trovato. Riprova a generare il video.")
    return jsonify({key: value for key, value in job.items() if key != "output"})


@app.get("/api/render/<job_id>/download")
def download_render(job_id):
    job = JOBS.get(job_id)
    if not job:
        render_status(job_id)
        job = JOBS.get(job_id)
    if not job or job.get("status") != "complete":
        abort(404, "Il video non è ancora pronto.")
    return send_file(job["output"], as_attachment=True, download_name="petcut-social-edit.mp4", mimetype="video/mp4")


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"File troppo grande: limite {MAX_UPLOAD_MB} MB."), 413


@app.errorhandler(HTTPException)
def api_http_error(error):
    """Keep API failures machine-readable so the interface can explain them."""
    if request.path.startswith("/api/"):
        return jsonify(error=error.description), error.code
    return error


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
