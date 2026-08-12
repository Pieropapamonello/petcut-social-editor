import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "exports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO = {"mp4", "mov", "m4v", "webm"}
ALLOWED_AUDIO = {"mp3", "wav", "m4a", "aac", "ogg"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def extension_allowed(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True)


def estimate_bpm(audio_path: Path) -> int:
    """Estimate a practical BPM from energy peaks; falls back cleanly for any audio."""
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
        energy = np.sqrt(np.mean(frames ** 2, axis=1))
        energy = np.maximum(energy - np.mean(energy), 0)
        if not np.any(energy):
            return 120
        correlation = np.correlate(energy, energy, mode="full")[len(energy) - 1 :]
        low = max(1, round((60 / 190) * rate / hop))
        high = min(len(correlation), round((60 / 70) * rate / hop))
        lag = low + int(np.argmax(correlation[low:high]))
        bpm = round(60 * rate / (hop * lag))
        return max(70, min(190, bpm))
    except Exception:
        return 120
    finally:
        wav_path.unlink(missing_ok=True)


def video_duration(path: Path) -> float:
    data = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)])
    return float(data.stdout.strip())


def visual_filter(style: str, bpm: int, title: str) -> str:
    frequency = bpm / 60
    saturation = {"cinematic": "1.18", "beat": "1.35", "collage": "1.25"}.get(style, "1.18")
    zoom = {"cinematic": "0.045", "beat": "0.080", "collage": "0.060"}.get(style, "0.045")
    filters = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
        "setsar=1",
        "fps=30",
        f"eq=contrast=1.10:saturation={saturation}:brightness=0.015",
        "unsharp=5:5:0.55:5:5:0.0",
        (
            f"zoompan=z='1.02+{zoom}*(0.5+0.5*sin(2*PI*{frequency:.4f}*on/30))'"
            ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30"
        ),
    ]
    if style == "collage":
        filters.append("vignette=PI/5")
    if title:
        clean_title = re.sub(r"[^A-Za-z0-9 !?'-]", "", title)[:32]
        if clean_title:
            filters.append(
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                f"text='{clean_title}':fontcolor=white:fontsize=72:borderw=4:bordercolor=black:"
                "x=(w-text_w)/2:y=h*0.15:enable='between(t,0,2.8)'"
            )
    return ",".join(filters)


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/api/health")
def health():
    return jsonify(status="ok", ffmpeg=shutil.which("ffmpeg") is not None)


@app.post("/api/render")
def render_video():
    video = request.files.get("video")
    audio = request.files.get("audio")
    style = request.form.get("style", "cinematic")
    title = request.form.get("title", "").strip()
    requested_seconds = min(60, max(5, int(request.form.get("duration", "15"))))
    if not video or not audio:
        abort(400, "Carica sia un video sia una canzone.")
    if not extension_allowed(video.filename, ALLOWED_VIDEO) or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Formato non supportato. Usa MP4/MOV/WebM e MP3/WAV/M4A/AAC/OGG.")
    if style not in {"cinematic", "beat", "collage"}:
        abort(400, "Preset non valido.")

    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="petcut-") as temp:
        temp_dir = Path(temp)
        video_path = temp_dir / f"input-video.{secure_filename(video.filename).rsplit('.', 1)[1]}"
        audio_path = temp_dir / f"input-audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
        video.save(video_path)
        audio.save(audio_path)
        duration = min(float(requested_seconds), video_duration(video_path), video_duration(audio_path))
        bpm = estimate_bpm(audio_path)
        destination = OUTPUT_DIR / f"petcut-{job_id}.mp4"
        command = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(video_path), "-i", str(audio_path),
            "-t", f"{duration:.2f}", "-filter:v", visual_filter(style, bpm, title),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(destination),
        ]
        try:
            run(command)
        except subprocess.CalledProcessError as error:
            app.logger.error(error.stderr)
            abort(500, "Non sono riuscito a generare il video. Prova un file diverso.")
    response = send_file(destination, as_attachment=True, download_name="petcut-social-edit.mp4", mimetype="video/mp4")
    response.headers["X-Detected-BPM"] = str(bpm)
    return response


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"File troppo grande: limite {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
