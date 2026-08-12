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
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}
ALLOWED_AUDIO = {"mp3", "wav", "m4a", "aac", "ogg"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


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
    beats_per_scene = {"cinematic": 4, "beat": 2, "collage": 2}.get(style, 3)
    return max(3, min(18, math.ceil(duration / ((60 / bpm) * beats_per_scene))))


def visual_filter(style: str, bpm: int, title: str) -> str:
    frequency = bpm / 60
    saturation = {"cinematic": "1.18", "beat": "1.35", "collage": "1.25"}[style]
    zoom = {"cinematic": "0.045", "beat": "0.080", "collage": "0.060"}[style]
    filters = [
        f"eq=contrast=1.10:saturation={saturation}:brightness=0.015",
        "unsharp=5:5:0.55:5:5:0.0",
        f"zoompan=z='1.02+{zoom}*(0.5+0.5*sin(2*PI*{frequency:.4f}*on/30))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30",
    ]
    if style == "collage":
        filters.append("vignette=PI/5")
    clean_title = re.sub(r"[^A-Za-z0-9 !?-]", "", title)[:32]
    if clean_title:
        filters.append("drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                       f"text='{clean_title}':fontcolor=white:fontsize=72:borderw=4:bordercolor=black:"
                       "x=(w-text_w)/2:y=h*0.15:enable='between(t,0,2.8)'")
    return ",".join(filters)


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


@app.post("/api/render")
def render_video():
    media = [file for file in request.files.getlist("media") if file and file.filename]
    audio = request.files.get("audio")
    style, title = request.form.get("style", "cinematic"), request.form.get("title", "").strip()
    requested_seconds = min(60, max(5, int(request.form.get("duration", "15"))))
    if not media or not audio:
        abort(400, "Carica almeno una foto o un video e una canzone.")
    if style not in {"cinematic", "beat", "collage"}:
        abort(400, "Preset non valido.")
    if any(not extension_allowed(file.filename, ALLOWED_VIDEO | ALLOWED_IMAGE) for file in media) or not extension_allowed(audio.filename, ALLOWED_AUDIO):
        abort(400, "Formato non supportato. Usa foto JPG/PNG/WebP, video MP4/MOV/WebM e audio MP3/WAV/M4A/AAC/OGG.")
    with tempfile.TemporaryDirectory(prefix="petcut-") as temp:
        temp_dir = Path(temp)
        audio_path = temp_dir / f"audio.{secure_filename(audio.filename).rsplit('.', 1)[1]}"
        audio.save(audio_path)
        duration, bpm = min(float(requested_seconds), media_duration(audio_path)), estimate_bpm(audio_path)
        scenes, scene_duration = recommended_scenes(duration, bpm, style), 0
        scene_duration = duration / scenes
        paths = []
        for index, file in enumerate(media):
            extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
            path = temp_dir / f"media-{index}.{extension}"
            file.save(path)
            paths.append(path)
        command = ["ffmpeg", "-y"]
        for path in paths:
            command.extend((["-loop", "1", "-framerate", "30", "-i", str(path)] if path.suffix[1:] in ALLOWED_IMAGE else ["-stream_loop", "-1", "-i", str(path)]))
        command.extend(["-i", str(audio_path)])
        labels, filters = [], []
        for index in range(scenes):
            label, input_index = f"s{index}", index % len(paths)
            filters.append(f"[{input_index}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30,trim=duration={scene_duration:.3f},setpts=PTS-STARTPTS[{label}]")
            labels.append(f"[{label}]")
        filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={scenes}:v=1:a=0[m];[m]{visual_filter(style, bpm, title)}[outv]"
        destination = OUTPUT_DIR / f"petcut-{uuid.uuid4().hex}.mp4"
        command.extend(["-t", f"{duration:.2f}", "-filter_complex", filter_complex, "-map", "[outv]", "-map", f"{len(paths)}:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
        try:
            run(command)
        except subprocess.CalledProcessError as error:
            app.logger.error(error.stderr)
            abort(500, "Non sono riuscito a generare il video. Prova un file diverso.")
    response = send_file(destination, as_attachment=True, download_name="petcut-social-edit.mp4", mimetype="video/mp4")
    response.headers["X-Detected-BPM"], response.headers["X-Recommended-Content"] = str(bpm), str(scenes)
    return response


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"File troppo grande: limite {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
