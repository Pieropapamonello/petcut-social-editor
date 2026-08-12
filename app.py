import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "exports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO = {"mp4", "mov", "m4v", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}
ALLOWED_AUDIO = {"mp3", "wav", "m4a", "aac", "ogg"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
RENDER_QUEUE = ThreadPoolExecutor(max_workers=1)


def extension_allowed(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True)


def media_duration(path: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)])
    return float(result.stdout.strip())


def estimate_bpm(audio_path: Path) -> int:
    wav_path = audio_path.with_suffix(".analysis.wav")
    try:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "22050", str(wav_path)])
        with wave.open(str(wav_path), "rb") as source:
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float32)
            rate = source.getframerate()
        if len(samples) < rate * 3:
            return 120
        hop = 1024
        frames = samples[: len(samples) // hop * hop].reshape(-1, hop)
        energy = np.maximum(np.sqrt(np.mean(frames**2, axis=1)) - np.mean(np.sqrt(np.mean(frames**2, axis=1))), 0)
        if not np.any(energy):
            return 120
        correlation = np.correlate(energy, energy, mode="full")[len(energy) - 1 :]
        low, high = max(1, round((60 / 190) * rate / hop)), min(len(correlation), round((60 / 70) * rate / hop))
        return max(70, min(190, round(60 * rate / (hop * (low + int(np.argmax(correlation[low:high])))))))
    except Exception:
        return 120
    finally:
        wav_path.unlink(missing_ok=True)


def recommended_scenes(duration: float, bpm: int, style: str) -> int:
    beats_per_scene = {"cinematic": 4, "beat": 2, "collage": 2, "floating": 3}.get(style, 3)
    return max(3, min(18, math.ceil(duration / ((60 / bpm) * beats_per_scene))))


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
        return Image.open(path).convert("RGBA")
    length = media_duration(path)
    timestamp = max(0.08, min(length - 0.08, length * (frame_number + 1) / (frame_count + 1)))
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return Image.open(BytesIO(result.stdout)).convert("RGBA")


def create_cutouts(paths: list[Path], job_dir: Path, count: int) -> list[Path]:
    """Lightweight foreground extraction for Render Free; subject should be centred."""
    cutouts = []
    for index in range(count):
        source = paths[index % len(paths)]
        frame = source_frame(source, index // len(paths), max(1, math.ceil(count / len(paths))))
        rgb = np.array(frame.convert("RGB"))
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        inset_x, inset_y = max(2, int(width * 0.06)), max(2, int(height * 0.04))
        foreground, background = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, mask, (inset_x, inset_y, width - inset_x * 2, height - inset_y * 2), foreground, background, 4, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        result = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
        bounds = result.getchannel("A").getbbox()
        if bounds:
            result = result.crop(bounds)
        result.thumbnail((520, 820), Image.Resampling.LANCZOS)
        destination = job_dir / f"cutout-{index}.png"
        result.save(destination)
        cutouts.append(destination)
    return cutouts


def floating_filter(cutouts: list[Path], raw_source: Path | None, duration: float, bpm: int, title: str) -> tuple[list[str], str]:
    """Build the black-background floating composition used by the reference edits."""
    floating_duration = duration if raw_source is None else max(3.5, duration * 0.72)
    scene_count = max(4, min(len(cutouts), recommended_scenes(floating_duration, bpm, "floating")))
    scene_duration = floating_duration / scene_count
    command = ["ffmpeg", "-y", "-f", "lavfi", "-t", f"{duration:.3f}", "-i", f"color=c=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r=30"]
    for cutout in cutouts:
        command.extend(["-loop", "1", "-framerate", "30", "-i", str(cutout)])
    raw_index = None
    if raw_source is not None:
        raw_index = len(cutouts) + 1
        command.extend(["-stream_loop", "-1", "-i", str(raw_source)])
    labels, filters = [], []
    positions = [("(main_w-overlay_w)/2", "(main_h-overlay_h)/2"), ("main_w*0.12", "main_h*0.17"), ("main_w*0.54", "main_h*0.43"), ("main_w*0.27", "main_h*0.52")]
    for index in range(scene_count):
        background, output = f"bg{index}", f"f{index}"
        filters.append(f"[0:v]trim=duration={scene_duration:.3f},setpts=PTS-STARTPTS[{background}]")
        overlay_base = background
        cutout_count = 1 if index < 2 else (2 if index % 3 else 3)
        for layer in range(cutout_count):
            cutout_index = (index + layer * 2) % len(cutouts) + 1
            width = 455 if layer == 0 else 260
            cut_label = f"c{index}_{layer}"
            x, y = positions[(index + layer) % len(positions)]
            filters.append(f"[{cutout_index}:v]format=rgba,scale={width}:-1,fade=t=in:st=0:d=0.12:alpha=1,fade=t=out:st={max(0.1, scene_duration - 0.10):.3f}:d=0.10:alpha=1[{cut_label}]")
            next_overlay = output if layer == cutout_count - 1 else f"o{index}_{layer}"
            filters.append(f"[{overlay_base}][{cut_label}]overlay=x='{x}':y='{y}':shortest=1:format=auto,setsar=1[{next_overlay}]")
            overlay_base = next_overlay
        labels.append(f"[{output}]")
    if raw_index is not None:
        reveal_duration = duration - floating_duration
        filters.append(f"[{raw_index}:v]trim=duration={reveal_duration:.3f},setpts=PTS-STARTPTS,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1,fps=30[reveal]")
        labels.append("[reveal]")
        scene_count += 1
    safe_title = re.sub(r"[^A-Za-z0-9 !?-]", "", title or "FLOATING")[:24].upper()
    filters.append("".join(labels) + f"concat=n={scene_count}:v=1:a=0[sequence]")
    filters.append(f"[sequence]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{safe_title}':fontcolor=white:fontsize=68:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h*0.77:enable='between(t,0.3,2.0)',eq=contrast=1.08:saturation=1.16,format=yuv420p[outv]")
    return command, ";".join(filters)


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/api/health")
def health():
    return jsonify(status="ok", ffmpeg=shutil.which("ffmpeg") is not None)


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
        bpm, usable_duration = estimate_bpm(path), min(float(requested_seconds), media_duration(path))
    scenes = recommended_scenes(usable_duration, bpm, style)
    return jsonify(bpm=bpm, recommended_content=scenes, message=(
        f"Ritmo rilevato: circa {bpm} BPM. Per un edit di {usable_duration:.0f} secondi, l'ideale sono {scenes} foto o clip. "
        "Puoi però caricarne anche una sola: PetCut la riutilizzerà creando movimenti e cambi sul beat."
    ))


def render_job(job_id, paths, audio_path, style, title, requested_seconds, job_dir):
    try:
        duration, bpm = min(float(requested_seconds), media_duration(audio_path)), estimate_bpm(audio_path)
        if style == "floating":
            cutout_count = max(5, min(9, recommended_scenes(duration, bpm, style)))
            cutouts = create_cutouts(paths, job_dir, cutout_count)
            raw_source = next((path for path in paths if path.suffix[1:] in ALLOWED_VIDEO), None)
            command, filter_complex = floating_filter(cutouts, raw_source, duration, bpm, title)
            destination = job_dir / "petcut-social-edit.mp4"
            audio_index = len(cutouts) + 1 + (1 if raw_source else 0)
            command.extend(["-i", str(audio_path), "-t", f"{duration:.2f}", "-filter_complex", filter_complex, "-map", "[outv]", "-map", f"{audio_index}:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
            run(command)
            JOBS[job_id].update(status="complete", output=str(destination), bpm=bpm, scenes=cutout_count)
            return
        scenes, scene_duration = recommended_scenes(duration, bpm, style), duration / recommended_scenes(duration, bpm, style)
        command = ["ffmpeg", "-y"]
        for path in paths:
            command.extend((["-loop", "1", "-framerate", "30", "-i", str(path)] if path.suffix[1:] in ALLOWED_IMAGE else ["-stream_loop", "-1", "-i", str(path)]))
        command.extend(["-i", str(audio_path)])
        labels, filters = [], []
        for index in range(scenes):
            label, input_index = f"s{index}", index % len(paths)
            source_length = 1 if paths[input_index].suffix[1:] in ALLOWED_IMAGE else media_duration(paths[input_index])
            start = 0 if source_length <= scene_duration else (source_length - scene_duration) * (index // len(paths)) / max(1, (scenes - 1) // len(paths))
            filters.append(f"[{input_index}:v]trim=start={start:.3f}:duration={scene_duration:.3f},setpts=PTS-STARTPTS,{scene_filter(index, style, bpm, scene_duration)}[{label}]")
            labels.append(f"[{label}]")
        filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={scenes}:v=1:a=0[m];[m]{final_filter(style, title)}[outv]"
        destination = job_dir / "petcut-social-edit.mp4"
        command.extend(["-t", f"{duration:.2f}", "-filter_complex", filter_complex, "-map", "[outv]", "-map", f"{len(paths)}:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
        run(command)
        JOBS[job_id].update(status="complete", output=str(destination), bpm=bpm, scenes=scenes)
    except Exception as error:
        app.logger.exception("Render failed")
        if isinstance(error, subprocess.CalledProcessError):
            app.logger.error(error.stderr)
        JOBS[job_id].update(status="failed", error="Generazione non riuscita. Prova un video più breve o riprova tra poco.")


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
