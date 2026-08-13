"""Memory-safe, reference-driven social video render engine.

The module deliberately renders one small H.264 clip at a time and joins the
clips with FFmpeg's concat demuxer.  This is slower than one enormous filter
graph, but it keeps peak memory predictable on Render's free instances.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


WIDTH, HEIGHT, FPS = 576, 1024, 30
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
Progress = Callable[[str, int], None]


def _run(command: Sequence[object]) -> subprocess.CompletedProcess:
    return subprocess.run([str(value) for value in command], capture_output=True, text=True, check=True)


def _notify(progress: Progress | None, stage: str, percent: int) -> None:
    if progress:
        progress(stage, max(0, min(100, int(percent))))


def _duration(path: Path) -> float:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", path])
    return max(0.1, float(result.stdout.strip()))


def _font_paths() -> list[str]:
    return [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]


def _font(size: int, condensed: bool = False) -> ImageFont.ImageFont:
    candidates = _font_paths() if condensed else _font_paths()[1:] + _font_paths()[:1]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _cover(image: Image.Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _open_rgba(path: Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGBA")


def _save(image: Image.Image, path: Path, quality: int = 94) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=quality, optimize=True)
    else:
        image.save(path)
    return path


def _fit_subject(subject: Image.Image, max_width: int, max_height: int) -> Image.Image:
    result = subject.copy()
    if result.width < 1 or result.height < 1:
        return result
    scale = min(max_width / result.width, max_height / result.height)
    size = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
    return result.resize(size, Image.Resampling.LANCZOS)


def _paste_with_rim(canvas: Image.Image, subject: Image.Image, xy: tuple[int, int], rim: bool = True) -> None:
    subject = subject.convert("RGBA")
    if rim and "A" in subject.getbands():
        # Draw only the one-pixel ring outside the matte. Applying the dilated
        # alpha itself used to wash the whole subject in a blue-grey halo.
        alpha = subject.getchannel("A")
        expanded = alpha.filter(ImageFilter.MaxFilter(3))
        ring = ImageChops.subtract(expanded, alpha).point(lambda value: int(value * 0.10))
        glow = Image.new("RGBA", subject.size, (226, 235, 246, 0))
        glow.putalpha(ring)
        canvas.alpha_composite(glow, xy)
    canvas.alpha_composite(subject, xy)


def _text_fit(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, condensed: bool = False) -> ImageFont.ImageFont:
    for size in range(start, 17, -2):
        font = _font(size, condensed=condensed)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        if box[2] - box[0] <= max_width:
            return font
    return _font(18, condensed=condensed)


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    max_width: int = WIDTH - 48,
    start_size: int = 82,
    fill: str = "white",
    stroke: int = 0,
    condensed: bool = False,
) -> None:
    font = _text_fit(draw, text, max_width, start_size, condensed)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    x = (WIDTH - (box[2] - box[0])) // 2
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill="black")


def _reference_cues(drop: float, beat: float, duration: float) -> dict[str, float | int]:
    """Scale the measured Download (5) cue sheet in musical beats.

    At 160 edit BPM these formulas reproduce the reference's exact act
    boundaries: F97, F255, F276 and F344.  A short pre-drop automatically
    uses two intro cards so the five-word phrase keeps its readable six-beat
    hold instead of being crushed into a few frames.
    """
    pre_drop_beats = drop / max(beat, 1 / FPS)
    intro_slots = 4 if pre_drop_beats >= 24.0 else 2
    intro_end = intro_slots * (97 / 45) * beat
    phrase_start = drop - (272 / 45) * beat
    roulette_end = phrase_start - (28 / 15) * beat

    # Preserve at least two beats of roulette and four frames for every word
    # on unusually short tracks while keeping all boundaries frame-exact.
    roulette_end = max(intro_end + 2 * beat, roulette_end)
    phrase_start = max(roulette_end + 4 / FPS, phrase_start)
    if phrase_start > drop - 15 / FPS:
        phrase_start = drop - 15 / FPS
        roulette_end = min(roulette_end, phrase_start - 4 / FPS)
    intro_end = min(intro_end, roulette_end - 4 / FPS)
    intro_end = max(8 / FPS, intro_end)
    roulette_end = max(intro_end + 4 / FPS, roulette_end)
    return {
        "intro_slots": intro_slots,
        "intro_end": round(intro_end * FPS) / FPS,
        "roulette_end": round(roulette_end * FPS) / FPS,
        "phrase_start": round(phrase_start * FPS) / FPS,
        "drop": round(min(duration - 0.8, drop) * FPS) / FPS,
    }


def _onsets(rhythm: dict, duration: float) -> list[float]:
    values = []
    for value in rhythm.get("onsets", []) or []:
        try:
            value = round(float(value) * FPS) / FPS
        except (TypeError, ValueError):
            continue
        if 0.0 < value < duration:
            values.append(value)
    return sorted(set(values))


def _strong_onsets(rhythm: dict, duration: float) -> list[float]:
    raw_onsets = rhythm.get("onsets", []) or []
    raw_strengths = rhythm.get("onset_strengths", []) or []
    if len(raw_onsets) != len(raw_strengths) or not raw_onsets:
        return []
    pairs = []
    for onset, strength in zip(raw_onsets, raw_strengths):
        try:
            onset, strength = float(onset), float(strength)
        except (TypeError, ValueError):
            continue
        if 0 < onset < duration and math.isfinite(strength):
            pairs.append((round(onset * FPS) / FPS, strength))
    if not pairs:
        return []
    threshold = float(np.percentile([strength for _, strength in pairs], 72))
    return [onset for onset, strength in pairs if strength >= threshold]


def _is_accent(time_value: float, strong_onsets: Sequence[float], radius: float = 0.10) -> bool:
    return any(abs(time_value - onset) <= radius for onset in strong_onsets)


def _accent_or_pattern(time_value: float, strong_onsets: Sequence[float], index: int, every: int) -> bool:
    return _is_accent(time_value, strong_onsets) or index % every == 0


def _beat(rhythm: dict) -> float:
    bpm = float(rhythm.get("edit_bpm") or rhythm.get("bpm") or 120)
    return 60.0 / max(70.0, min(200.0, bpm))


def _snap(value: float, onsets: Sequence[float], radius: float = 0.11) -> float:
    nearby = [onset for onset in onsets if abs(onset - value) <= radius]
    result = min(nearby, key=lambda onset: abs(onset - value)) if nearby else value
    return round(result * FPS) / FPS


def _drop_time(rhythm: dict, duration: float, fallback: float) -> float:
    try:
        candidate = float(rhythm.get("drop_time"))
    except (TypeError, ValueError):
        candidate = fallback
    if not 1.0 <= candidate <= duration - 1.0:
        candidate = fallback
    return _snap(candidate, _onsets(rhythm, duration), 0.18)


def _visual_drop_time(rhythm: dict, duration: float, fallback: float) -> float:
    """Find the first strong impact that starts the reference's drop cluster."""
    energy_drop = _drop_time(rhythm, duration, fallback)
    beat = _beat(rhythm)
    target = energy_drop - 1.5 * beat
    candidates = []
    for onset, strength in zip(rhythm.get("onsets", []) or [], rhythm.get("onset_strengths", []) or []):
        try:
            onset, strength = round(float(onset) * FPS) / FPS, float(strength)
        except (TypeError, ValueError):
            continue
        if strength >= 0.72 and abs(onset - target) <= 0.16:
            candidates.append(onset)
    if candidates:
        return min(candidates, key=lambda value: abs(value - target))
    return energy_drop


def _frame_boundaries(start: float, end: float, count: int, onsets: Sequence[float], minimum: int = 4) -> list[int]:
    start_frame, end_frame = round(start * FPS), round(end * FPS)
    count = max(1, min(count, max(1, (end_frame - start_frame) // minimum)))
    onset_frames = [round(onset * FPS) for onset in onsets if start < onset < end]
    result = [start_frame]
    for index in range(1, count):
        target = round(start_frame + (end_frame - start_frame) * index / count)
        low = result[-1] + minimum
        high = end_frame - minimum * (count - index)
        candidates = [frame for frame in onset_frames if low <= frame <= high and abs(frame - target) <= 4]
        result.append(min(candidates, key=lambda frame: abs(frame - target)) if candidates else max(low, min(high, target)))
    result.append(end_frame)
    return result


def _durations(start: float, end: float, count: int, onsets: Sequence[float], minimum: int = 4) -> list[float]:
    boundaries = _frame_boundaries(start, end, count, onsets, minimum)
    return [(second - first) / FPS for first, second in zip(boundaries, boundaries[1:])]


def _weighted_frame_durations(start: float, end: float, weights: Sequence[float], minimum: int = 1) -> list[float]:
    """Allocate an exact frame span with the largest-remainder method."""
    span = max(len(weights) * minimum, round(end * FPS) - round(start * FPS))
    values = np.asarray(weights, dtype=np.float64)
    values = np.maximum(values, 1e-6)
    raw = values / values.sum() * span
    frames = np.maximum(minimum, np.floor(raw).astype(int))
    while int(frames.sum()) < span:
        index = int(np.argmax(raw - frames))
        frames[index] += 1
    while int(frames.sum()) > span:
        eligible = np.flatnonzero(frames > minimum)
        if not len(eligible):
            break
        index = int(eligible[np.argmax(frames[eligible] - raw[eligible])])
        frames[index] -= 1
    return [int(value) / FPS for value in frames]


def _onset_aligned_weighted_durations(
    start: float,
    end: float,
    weights: Sequence[float],
    onsets: Sequence[float],
    minimum: int = 4,
    radius_frames: int = 3,
) -> list[float]:
    """Keep a weighted cue sheet while moving inner cuts onto transients."""
    base = [round(value * FPS) for value in _weighted_frame_durations(start, end, weights, minimum)]
    start_frame, end_frame = round(start * FPS), round(end * FPS)
    onset_frames = sorted({round(value * FPS) for value in onsets if start < value < end})
    boundaries = [start_frame]
    target = start_frame
    for index, frame_count in enumerate(base[:-1], start=1):
        target += frame_count
        low = boundaries[-1] + minimum
        high = end_frame - minimum * (len(base) - index)
        nearby = [frame for frame in onset_frames if low <= frame <= high and abs(frame - target) <= radius_frames]
        boundaries.append(min(nearby, key=lambda frame: abs(frame - target)) if nearby else max(low, min(high, target)))
    boundaries.append(end_frame)
    return [(right - left) / FPS for left, right in zip(boundaries, boundaries[1:])]


def _dense_durations(start: float, end: float, target_frames: float = 3.1, maximum: int = 64) -> list[float]:
    span = max(1, round(end * FPS) - round(start * FPS))
    count = max(1, min(maximum, round(span / target_frames)))
    return _weighted_frame_durations(start, end, [1.0] * count, minimum=2)


def _section_boundary(rhythm: dict, name: str, duration: float) -> float | None:
    sections = rhythm.get("sections") or []
    for section in sections:
        if isinstance(section, dict) and str(section.get("name", "")).lower() == name:
            for key in ("start", "time"):
                try:
                    value = float(section[key])
                    if 0 < value < duration:
                        return value
                except (KeyError, TypeError, ValueError):
                    pass
    return None


def _sample_video_candidates(path: Path, limit: int = 24) -> list[tuple[float, float]]:
    """Return diverse, sharp, moving source times without retaining frames."""
    capture = cv2.VideoCapture(str(path))
    length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    seconds = length / fps if length > 0 else _duration(path)
    sample_count = min(90, max(18, int(seconds * 4)))
    times = np.linspace(0.05, max(0.06, seconds - 0.08), sample_count)
    records: list[tuple[float, float, np.ndarray]] = []
    previous = None
    for time_value in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_value * 1000))
        ok, frame = capture.read()
        if not ok:
            continue
        small = cv2.resize(frame, (144, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = min(1.0, cv2.Laplacian(gray, cv2.CV_32F).var() / 900.0)
        motion = 0.0 if previous is None else min(1.0, cv2.absdiff(gray, previous).mean() / 28.0)
        luminance = float(gray.mean())
        exposure = max(0.0, 1.0 - abs(luminance - 128.0) / 128.0)
        score = 0.46 * sharpness + 0.38 * motion + 0.16 * exposure
        records.append((score, float(time_value), gray))
        previous = gray
    capture.release()
    def difference_hash(gray: np.ndarray) -> np.ndarray:
        tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        return (tiny[:, 1:] > tiny[:, :-1]).reshape(-1)

    pool = [(score, time_value, difference_hash(gray)) for score, time_value, gray in records]
    selected: list[tuple[float, float, np.ndarray]] = []
    minimum_gap = max(0.18, seconds / max(12, limit * 3))
    while pool and len(selected) < limit:
        if not selected:
            chosen = max(pool, key=lambda value: value[0])
        else:
            def diversity(value: tuple[float, float, np.ndarray]) -> float:
                score, time_value, visual_hash = value
                visual_distance = min(np.count_nonzero(visual_hash != prior_hash) / 64 for _, _, prior_hash in selected)
                time_distance = min(abs(time_value - prior_time) / max(1.0, seconds) for _, prior_time, _ in selected)
                return 0.55 * score + 0.27 * visual_distance + 0.18 * min(1.0, time_distance * 4)

            candidates = [
                value for value in pool
                if all(abs(value[1] - prior_time) >= minimum_gap for _, prior_time, _ in selected)
            ]
            if not candidates:
                break
            chosen = max(candidates, key=diversity)
        selected.append(chosen)
        pool.remove(chosen)
    if not selected:
        return [(0.0, 0.0)]
    # Keep the greedy MMR order: consecutive clips then remain as different
    # as the source permits instead of returning to chronological similarity.
    return [(time_value, score) for score, time_value, _ in selected]


def _extract_frame(path: Path, time_value: float, destination: Path) -> Path:
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{time_value:.3f}", "-i", path, "-frames:v", "1", "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}", "-q:v", "2", destination])
    return destination


def _source_assets(paths: Sequence[Path], samples: Sequence[Path] | None, job_dir: Path) -> tuple[list[Path], list[tuple[Path, float]]]:
    images = [Path(value) for value in (samples or []) if Path(value).exists()]
    video_groups: list[list[tuple[Path, float]]] = []
    for path_value in paths:
        path = Path(path_value)
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
            images.append(path)
        elif path.suffix.lower() in VIDEO_EXTENSIONS and path.exists():
            video_groups.append([(path, time_value) for time_value, _ in _sample_video_candidates(path)])
    # Round-robin different uploads before taking another moment from the
    # same clip. This makes four supplied animals/people map naturally to the
    # four reference intro cards and avoids exhausting one source in climax.
    video_times = [
        group[index]
        for index in range(max((len(group) for group in video_groups), default=0))
        for group in video_groups
        if index < len(group)
    ]
    # Samples supplied by the old app are preferred; otherwise create a few
    # reusable full-frame stills from the scored video moments.
    if not images:
        for index, (path, time_value) in enumerate(video_times[:8]):
            images.append(_extract_frame(path, time_value, job_dir / f"engine-source-{index:02d}.jpg"))
    return images, video_times


def _auto_cutout(source: Path, destination: Path) -> Path:
    """Create a conservative centre-subject matte when the caller has none."""
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image source {source}")
    height, width = bgr.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = bgr.shape[:2]
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    border_x, border_y = max(2, width // 25), max(2, height // 30)
    mask[:border_y] = cv2.GC_BGD
    mask[-border_y:] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, -border_x:] = cv2.GC_BGD
    # The central ellipse is probable foreground, not hard foreground, so the
    # model may still remove floor/wall pixels inside it.
    centre = np.zeros_like(mask)
    cv2.ellipse(centre, (width // 2, height // 2), (max(8, width * 36 // 100), max(8, height * 44 // 100)), 0, 0, 360, 255, -1)
    mask[centre > 0] = cv2.GC_PR_FGD
    sure = np.zeros_like(mask)
    cv2.ellipse(sure, (width // 2, height // 2), (max(4, width // 8), max(4, height // 6)), 0, 0, 360, 255, -1)
    mask[sure > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, background_model, foreground_model, 2, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        alpha = centre
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), rgba)
    return destination


def _cutout_assets(cutouts: Sequence[Path] | None, images: Sequence[Path], job_dir: Path) -> list[Path]:
    valid = [Path(path) for path in (cutouts or []) if Path(path).exists()]
    if valid:
        return valid
    generated = []
    for index, source in enumerate(images[:8]):
        generated.append(_auto_cutout(source, job_dir / f"engine-auto-cutout-{index:02d}.png"))
    return generated


def _still_clip(image_path: Path, destination: Path, duration: float, mode: str, accent: bool = False, index: int = 0) -> None:
    frames = max(1, round(duration * FPS))
    last = max(1, frames - 1)
    impact = mode.startswith("impact-")
    if impact:
        mode = mode.removeprefix("impact-")
    if mode == "hold":
        # Even the readable holds in the references breathe by a few pixels.
        # Keep it subtle so the subject is stable, but never a frozen JPEG.
        z = f"1.012+0.014*sin(PI*on/{last})"
        x, y = f"iw/2-(iw/zoom/2)+3*sin(2*PI*on/{last})", f"ih/2-(ih/zoom/2)-3*sin(PI*on/{last})"
    elif mode in {"push", "intro-out"}:
        z = f"1+0.10*(on/{last})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "intro-in":
        # Raw cards in the reference begin on the source image, never on a
        # full white frame.  A two-frame punch supplies the entrance energy.
        z = f"1.055-0.045*min(1\\,on/2)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "pull":
        z = f"1.10-0.10*(on/{last})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "whip-left":
        z = "1.08"
        x, y = f"(iw-iw/zoom)*(on/{last})", "ih/2-(ih/zoom/2)"
    elif mode == "whip-right":
        z = "1.08"
        x, y = f"(iw-iw/zoom)*(1-on/{last})", "ih/2-(ih/zoom/2)"
    elif mode == "roulette":
        # Download (5) gets its energy from a different pose every 3–4
        # frames.  Animating inside such a tiny shot made PetCut look like a
        # continuous mechanical zoom instead of a clean roulette.
        z, x, y = "1.0", "0", "0"
    elif mode == "roulette-entry":
        z = f"1.09-0.09*min(1\\,on/6)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "collage":
        z = f"1.005+0.028*sin(PI*on/{last})"
        x = f"iw/2-(iw/zoom/2)+4*sin(2*PI*on/{last})"
        y = f"ih/2-(ih/zoom/2)-3*sin(PI*on/{last})"
    else:
        z = f"1+0.05*sin(PI*on/{last})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if mode == "hard":
        z, x, y = "1.0", "0", "0"
    filters = [f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}"]
    edge = min(0.067, duration * 0.18)
    if mode.startswith("whip"):
        filters.append(
            f"gblur=sigma=8:enable='lt(t,{edge:.3f})+gt(t,{max(0.0, duration - edge):.3f})'"
        )
    if mode == "intro-out":
        transition = min(0.10, max(1 / FPS, duration * 0.28))
        filters.append(f"gblur=sigma=7:enable='gt(t,{max(0.0, duration - transition):.3f})'")
        filters.append(f"fade=t=out:st={max(0.0, duration - transition):.3f}:d={transition:.3f}:color=white")
    elif mode == "intro-in":
        filters.append("gblur=sigma=3:enable='eq(n,0)'")
    if mode == "roulette-entry":
        filters.extend([
            "gblur=sigma=8:enable='lt(n,4)'",
            "drawbox=color=white@0.88:t=fill:enable='eq(n,0)'",
            "drawbox=color=white@0.72:t=fill:enable='eq(n,1)'",
            "drawbox=color=white@0.55:t=fill:enable='eq(n,2)'",
            "drawbox=color=white@0.39:t=fill:enable='eq(n,3)'",
            "drawbox=color=white@0.24:t=fill:enable='eq(n,4)'",
            "drawbox=color=white@0.10:t=fill:enable='eq(n,5)'",
        ])
    if impact:
        filters.extend([
            "gblur=sigma=7:enable='lt(n,2)'",
            "drawbox=color=white@0.82:t=fill:enable='eq(n,0)'",
            "drawbox=color=white@0.66:t=fill:enable='eq(n,1)'",
            "drawbox=color=white@0.49:t=fill:enable='eq(n,2)'",
            "drawbox=color=white@0.32:t=fill:enable='eq(n,3)'",
            "drawbox=color=white@0.17:t=fill:enable='eq(n,4)'",
            "drawbox=color=white@0.07:t=fill:enable='eq(n,5)'",
        ])
    elif accent:
        filters.extend([
            "drawbox=color=white@0.56:t=fill:enable='eq(n,0)'",
            "drawbox=color=white@0.22:t=fill:enable='eq(n,1)'",
        ])
    _run(["ffmpeg", "-y", "-loop", "1", "-framerate", FPS, "-i", image_path, "-frames:v", frames, "-vf", ",".join(filters), "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1", "-pix_fmt", "yuv420p", "-r", FPS, "-g", 15, "-keyint_min", 1, "-sc_threshold", 0, destination])


def _video_clip(
    source: Path,
    destination: Path,
    start: float,
    duration: float,
    mode: str,
    accent: bool = False,
    index: int = 0,
) -> None:
    frames = max(1, round(duration * FPS))
    last = max(1, frames - 1)
    direction = -1 if index % 2 else 1
    timing = "PTS/1.35" if mode == "speed" else ("PTS/1.16" if mode in {"beat", "whip", "punch"} else "PTS")
    scale_filter = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}"
    if mode in {"whip", "punch", "pull", "detail", "hflip"}:
        scale_filter = "scale=620:1102:force_original_aspect_ratio=increase,crop=576:1024"
    elif mode == "close-left":
        scale_filter = "scale=720:1280:force_original_aspect_ratio=increase,crop=576:1024:0:(in_h-out_h)/2"
    elif mode == "close-right":
        scale_filter = "scale=720:1280:force_original_aspect_ratio=increase,crop=576:1024:in_w-out_w:(in_h-out_h)/2"
    elif mode == "close-high":
        scale_filter = "scale=720:1280:force_original_aspect_ratio=increase,crop=576:1024:(in_w-out_w)/2:0"
    elif mode == "close-low":
        scale_filter = "scale=720:1280:force_original_aspect_ratio=increase,crop=576:1024:(in_w-out_w)/2:in_h-out_h"
    filters = [f"setpts='{timing}'", scale_filter]
    if mode == "hflip":
        filters.append("hflip")
    exit_start = max(0, last - 4)
    if mode == "beat":
        filters.append(f"zoompan=z='1.00+0.40*(on/{last})':x='iw/2-(iw/zoom/2)+{direction}*54*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)-38*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    elif mode == "whip":
        filters.append(f"zoompan=z='1.02+0.18*(1-min(1\\,on/3))+0.36*(on/{last})+0.12*max(0\\,(on-{exit_start})/4)':x='iw/2-(iw/zoom/2)+{direction}*76*(1-min(1\\,on/3))-{direction}*62*max(0\\,(on-{exit_start})/4)':y='ih/2-(ih/zoom/2)-34*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
        filters.append(f"gblur=sigma=7:enable='eq(n,0)+gte(n,{last})'")
    elif mode == "punch":
        filters.append(f"zoompan=z='1.03+0.23*(1-min(1\\,on/3))+0.38*(on/{last})+0.13*max(0\\,(on-{exit_start})/4)':x='iw/2-(iw/zoom/2)+{direction}*38*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)-22*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
        filters.append("gblur=sigma=6:enable='eq(n,0)'")
    elif mode == "pull":
        filters.append(f"zoompan=z='1.42-0.34*min(1\\,on/4)+0.20*(on/{last})':x='iw/2-(iw/zoom/2)-{direction}*54*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)+34*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    elif mode == "detail":
        filters.append(f"zoompan=z='1.08+0.36*(on/{last})':x='iw/2-(iw/zoom/2)+{direction}*58*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)-38+40*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    elif mode == "hflip":
        filters.append(f"zoompan=z='1.02+0.38*(on/{last})':x='iw/2-(iw/zoom/2)-{direction}*54*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)-36*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    elif mode.startswith("close-"):
        filters.append(f"zoompan=z='1.01+0.36*(on/{last})':x='iw/2-(iw/zoom/2)+{direction}*54*(on/{last})':y='ih/2-(ih/zoom/2)-38*sin(PI*on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    elif mode == "speed":
        filters.append(f"zoompan=z='1.01+0.40*(on/{last})':x='iw/2-(iw/zoom/2)+{direction}*48*sin(PI*on/{last})':y='ih/2-(ih/zoom/2)-48*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    else:
        filters.append(f"fps={FPS}")
    filters.extend(["eq=gamma=1.20:contrast=1.07:saturation=1.08:brightness=0.015", "unsharp=5:5:0.48"])
    if accent:
        filters.extend([
            "drawbox=color=white@0.68:t=fill:enable='eq(n,0)'",
            "drawbox=color=white@0.30:t=fill:enable='eq(n,1)'",
        ])
        if frames >= 8 and index % 2 == 0:
            filters.append(f"drawbox=color=white@0.24:t=fill:enable='eq(n,{frames // 2})'")
    _run(["ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{max(0.0, start):.3f}", "-i", source, "-frames:v", frames, "-vf", ",".join(filters), "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1", "-pix_fmt", "yuv420p", "-r", FPS, "-g", 15, "-keyint_min", 1, "-sc_threshold", 0, destination])


def _intro_video_clip(
    source: Path,
    overlay: Path,
    destination: Path,
    start: float,
    duration: float,
) -> None:
    """Render a live raw card with typography over the moving source."""
    frames = max(1, round(duration * FPS))
    last = max(1, frames - 1)
    filters = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},zoompan=z='1.01+0.10*(on/{last})':"
        f"x='iw/2-(iw/zoom/2)+14*sin(PI*on/{last})':"
        f"y='ih/2-(ih/zoom/2)-10*(on/{last})':d=1:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"eq=contrast=1.04:saturation=1.12,"
        f"unsharp=5:5:0.35[base];"
        "[base][1:v]overlay=0:0:format=auto,format=yuv420p[out]"
    )
    _run([
        "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-ss", f"{max(0.0, start):.3f}", "-i", source,
        "-loop", "1", "-framerate", FPS, "-i", overlay, "-frames:v", frames,
        "-filter_complex", filters, "-map", "[out]", "-an", "-c:v", "libx264",
        "-preset", "ultrafast", "-crf", "22", "-threads", "1", "-pix_fmt", "yuv420p",
        "-r", FPS, "-g", 15, "-keyint_min", 1, "-sc_threshold", 0, destination,
    ])


def _finish(clips: Sequence[Path], audio_path: Path, destination: Path, duration: float, job_dir: Path) -> None:
    if not clips:
        raise ValueError("No clips were rendered")
    concat_path = job_dir / "engine-clips.txt"
    concat_path.write_text("".join(f"file '{clip.resolve().as_posix()}'\n" for clip in clips), encoding="utf-8")
    # Normalise the tiny concat inputs into one CFR, limited-range BT.709
    # stream.  Copying the first bitstream header leaked JPEG/full-range
    # metadata into some exports and visibly lifted the black backgrounds.
    _run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_path,
        "-i", audio_path, "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "1",
        "-pix_fmt", "yuv420p", "-r", FPS, "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-c:a", "aac", "-b:a", "192k",
        "-ar", "44100", "-movflags", "+faststart", destination,
    ])


class _Timeline:
    def __init__(self, job_dir: Path, audio_path: Path, destination: Path, duration: float, progress: Progress | None):
        self.job_dir = job_dir
        self.audio_path = audio_path
        self.destination = destination
        self.duration = duration
        self.progress = progress
        self.clips: list[Path] = []

    def path(self) -> Path:
        return self.job_dir / f"engine-clip-{len(self.clips):03d}.mp4"

    def add_still(self, board: Path, clip_duration: float, mode: str = "hold", accent: bool = False) -> None:
        target = self.path()
        _still_clip(board, target, clip_duration, mode, accent, len(self.clips))
        self.clips.append(target)

    def add_video(self, source: Path, start: float, clip_duration: float, mode: str = "beat", accent: bool = False) -> None:
        target = self.path()
        _video_clip(source, target, start, clip_duration, mode, accent, len(self.clips))
        self.clips.append(target)

    def add_intro_video(self, source: Path, overlay: Path, start: float, clip_duration: float) -> None:
        target = self.path()
        _intro_video_clip(source, overlay, target, start, clip_duration)
        self.clips.append(target)

    def finish(self) -> None:
        _notify(self.progress, "finalizzazione", 94)
        _finish(self.clips, self.audio_path, self.destination, self.duration, self.job_dir)


def _raw_board(
    source: Path,
    label: str,
    destination: Path,
    darken: float = 0.0,
) -> Path:
    canvas = _cover(Image.open(source)).convert("RGBA")
    if darken:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, int(255 * darken)))
        canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    # Keep the name fully readable on top of the live/raw card. The source
    # already contains the animal, so compositing a second subject here would
    # duplicate it and hide nearly the whole word.
    intro_text = label.upper()
    intro_font = _text_fit(draw, intro_text, WIDTH - 48, 112)
    box = draw.textbbox((0, 0), intro_text, font=intro_font)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, int(HEIGHT * 0.445)), intro_text, font=intro_font, fill="white")
    return _save(canvas, destination)


def _intro_text_overlay(label: str, destination: Path) -> Path:
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    text = label.upper()
    font = _text_fit(draw, text, WIDTH - 48, 112)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, int(HEIGHT * 0.445)), text, font=font, fill="white")
    return _save(canvas, destination)


def _cutout_board(subject_path: Path, destination: Path, index: int, label: str | None = None) -> Path:
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), "black")
    sizes = ((260, 420), (320, 500), (280, 455), (340, 525))
    subject = _fit_subject(_open_rgba(subject_path), *sizes[index % len(sizes)])
    if index % 5 == 2:
        subject = ImageOps.mirror(subject)
    offsets = [(-95, -40), (70, -22), (-32, 48), (92, 28), (8, -52), (-68, 42)]
    ox, oy = offsets[index % len(offsets)]
    xy = ((WIDTH - subject.width) // 2 + ox, (HEIGHT - subject.height) // 2 + oy)
    _paste_with_rim(canvas, subject, xy, rim=False)
    if label:
        _center_text(ImageDraw.Draw(canvas), label.upper(), int(HEIGHT * 0.67), start_size=78)
    return _save(canvas, destination)


def _collage_board(subject_paths: Sequence[Path], destination: Path, word: str, count: int, layout_seed: int = 0) -> Path:
    """Place text first, then subjects, creating the reference's text-behind z-order."""
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(canvas)
    # Text sits in the negative-space corridor and is deliberately covered
    # by the four large subjects, matching the reference's layer order.
    text = word.upper()
    if text:
        measured_sizes = {"CAN": 146, "YOU": 146, "IMAGINE": 108, "FLOATING": 96, "WEIGHTLESS": 80}
        font = _text_fit(draw, text, WIDTH - 46, measured_sizes.get(text, 108))
        box = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        text_x = (WIDTH - text_width) / 2 - box[0]
        text_y = 548 - text_height / 2 - box[1]
        draw.text((text_x, text_y), text, font=font, fill="white")
    positions = [(-18, 84), (288, 104), (-22, 560), (286, 576), (126, 70), (128, 600)]
    for index in range(max(1, count)):
        subject = _fit_subject(_open_rgba(subject_paths[(index * 3 + layout_seed) % len(subject_paths)]), 330, 440)
        if index % 3 == 1:
            subject = ImageOps.mirror(subject)
        x, y = positions[index % len(positions)]
        _paste_with_rim(canvas, subject, (x, y), rim=False)
    return _save(canvas, destination)


def _intro_labels(title: str) -> list[str]:
    values = [value.strip().upper() for value in title.replace("|", ",").replace("/", ",").split(",") if value.strip()]
    if values:
        return [values[index % len(values)] for index in range(4)]
    return ["PET"] * 4


def _intro_asset_groups(
    paths: Sequence[Path], images: Sequence[Path], cutouts: Sequence[Path],
) -> list[tuple[int, list[tuple[Path, Path]]]]:
    """Keep every intro raw frame paired with a cutout from its source.

    ``create_cutouts`` returns selected samples and cutouts in the same order;
    ``_source_assets`` keeps those samples at the beginning of ``images``.
    Grouping that paired prefix prevents a card for one upload from being
    followed by the pose of another when a weak matte has been filtered out.
    """
    groups: dict[int, list[tuple[Path, Path]]] = {}
    for pair_index, (source_image, cutout) in enumerate(zip(images, cutouts)):
        source_match = re.fullmatch(r"sample-\d+-source-(\d+)", source_image.stem)
        source_index = int(source_match.group(1)) if source_match else pair_index % len(paths)
        groups.setdefault(source_index % len(paths), []).append((Path(source_image), Path(cutout)))
    return list(groups.items())


def _render_reference_edit(
    paths: Sequence[Path], images: list[Path], video_times: list[tuple[Path, float]], cutouts: list[Path], audio: Path,
    job_dir: Path, destination: Path, duration: float, rhythm: dict, title: str, progress: Progress | None,
) -> dict:
    onsets, strong, beat = _onsets(rhythm, duration), _strong_onsets(rhythm, duration), _beat(rhythm)
    fallback_drop = duration * (11.47 / 24.23)
    drop = _visual_drop_time(rhythm, duration, fallback_drop)
    # Very early analyzer candidates cannot contain the measured intro,
    # roulette and five-word cue sheet. Keep short edits frame-safe instead
    # of over-allocating those sections and truncating the final climax.
    minimum_drop = duration * (0.50 if duration < 8 else 0.40)
    drop = round(max(minimum_drop, min(duration - 0.8, drop)) * FPS) / FPS
    cues = _reference_cues(drop, beat, duration)
    drop = float(cues["drop"])
    intro_end = float(cues["intro_end"])
    roulette_end = float(cues["roulette_end"])
    phrase_start = float(cues["phrase_start"])
    timeline = _Timeline(job_dir, audio, destination, duration, progress)
    labels = _intro_labels(title)
    intro_sources = _intro_asset_groups(paths, images, cutouts)

    # Act 1: each subject gets raw context followed by a cutout pose.
    _notify(progress, "intro", 8)
    intro_slots = int(cues["intro_slots"])
    slot_weights = [29, 23, 23, 22][:intro_slots]
    slot_durations = _weighted_frame_durations(0, intro_end, slot_weights, minimum=8)
    for index, slot_duration in enumerate(slot_durations):
        slot_frames = max(4, round(slot_duration * FPS))
        raw_frames = max(2, min(slot_frames - 2, round(slot_frames * 0.55)))
        raw_duration, cut_duration = raw_frames / FPS, (slot_frames - raw_frames) / FPS
        source_index, source_assets = intro_sources[index % len(intro_sources)]
        variant_index = (index // len(intro_sources)) % len(source_assets)
        source_image, _ = source_assets[variant_index]
        label = labels[source_index % len(labels)]
        original_source = Path(paths[source_index])
        slot_video_times = [value for value in video_times if value[0] == original_source]
        if original_source.suffix.lower() in VIDEO_EXTENSIONS and slot_video_times:
            source_video, source_time = slot_video_times[(index // max(1, len(paths))) % len(slot_video_times)]
            overlay = _intro_text_overlay(label, job_dir / f"reference-intro-text-{index:02d}.png")
            timeline.add_intro_video(source_video, overlay, source_time, raw_duration)
        else:
            board = _raw_board(
                source_image,
                label,
                job_dir / f"reference-intro-raw-{index:02d}.jpg",
                darken=0.04,
            )
            timeline.add_still(board, raw_duration, "intro-in", accent=False)

        # The reference fires several 3-frame poses after every raw card.
        pose_count = max(1, min(4, round((slot_frames - raw_frames) / 3)))
        pose_durations = _weighted_frame_durations(0, cut_duration, [1] * pose_count, minimum=2)
        for pose_index, pose_duration in enumerate(pose_durations):
            cutout_path = source_assets[(variant_index + pose_index) % len(source_assets)][1]
            board = _cutout_board(
                cutout_path,
                job_dir / f"reference-intro-cut-{index:02d}-{pose_index:02d}.png",
                index * 3 + pose_index,
            )
            timeline.add_still(
                board,
                pose_duration,
                "roulette",
                accent=False,
            )

    # Act 2: the measured reference cadence is mostly three frames per pose,
    # with a hard cap of 48 poses for a 24-second edit.
    _notify(progress, "roulette", 28)
    roulette_span_frames = round((roulette_end - intro_end) * FPS)
    entry_frames = min(max(2, roulette_span_frames - 2), max(2, round((44 / 45) * beat * FPS)))
    entry_duration = entry_frames / FPS
    entry_board = _cutout_board(cutouts[0], job_dir / "reference-roulette-entry.png", 0)
    timeline.add_still(entry_board, entry_duration, "roulette-entry", accent=False)
    roulette_micro_start = intro_end + entry_duration
    roulette_durations = _dense_durations(roulette_micro_start, roulette_end, 3.06, maximum=48)
    for index, clip_duration in enumerate(roulette_durations):
        board = _cutout_board(cutouts[(index * 7 + index // max(1, len(cutouts))) % len(cutouts)], job_dir / f"reference-roulette-{index:02d}.png", index + 1)
        timeline.add_still(board, clip_duration, "roulette", accent=False)

    # Act 3: subjects first accumulate around an empty centre, then the five
    # words use the measured unequal holds from the reference.
    _notify(progress, "collage", 48)
    build_count = max(2, min(4, round((phrase_start - roulette_end) / max(0.14, beat * 0.62))))
    build_durations = _weighted_frame_durations(roulette_end, phrase_start, [1] * build_count, minimum=4)
    for index, clip_duration in enumerate(build_durations):
        board = _collage_board(
            cutouts,
            job_dir / f"reference-collage-build-{index:02d}.png",
            "",
            index + 2,
            0,
        )
        timeline.add_still(board, clip_duration, "collage", accent=False)

    words = ["CAN", "YOU", "IMAGINE", "FLOATING", "WEIGHTLESS"]
    collage_durations = _weighted_frame_durations(phrase_start, drop, [7, 7, 22, 13, 19], minimum=3)
    for index, clip_duration in enumerate(collage_durations):
        board = _collage_board(cutouts, job_dir / f"reference-collage-{index:02d}.png", words[index], 4, 0)
        timeline.add_still(board, clip_duration, "collage", accent=False)

    # Act 4: alternate beat and half-beat source clips. Scored moments avoid
    # dead/off-focus footage; accents are short and sparse.
    _notify(progress, "climax", 62)
    remaining = duration - drop
    climax_pattern = [6, 6, 5, 6, 11, 12, 7, 4, 6, 5, 11, 17, 10, 7, 11, 12, 23, 5, 7, 4, 6, 16, 4, 13, 11, 6, 12, 5, 11, 17, 11, 6, 11, 12, 11, 6, 5, 11, 15, 6, 13]
    impact_frames = [6, 6, 5, 6]
    available_frames = max(0, round(remaining * FPS))
    while impact_frames and sum(impact_frames) > max(0, available_frames - 4):
        impact_frames.pop()
    impact_durations = [frames / FPS for frames in impact_frames]
    raw_start = drop + sum(impact_durations)
    raw_remaining = max(0.0, duration - raw_start)
    raw_target_count = max(1, min(60, round(raw_remaining / max(4 / FPS, beat * 0.865)))) if raw_remaining >= 4 / FPS else 0
    raw_weights = [climax_pattern[(index + 4) % len(climax_pattern)] for index in range(raw_target_count)]
    raw_durations = (
        _onset_aligned_weighted_durations(raw_start, duration, raw_weights, onsets, minimum=4, radius_frames=2)
        if raw_weights else []
    )
    for index, clip_duration in enumerate(impact_durations):
        board = _collage_board(
            cutouts,
            job_dir / f"reference-drop-impact-{index:02d}.png",
            "",
            min(6, 4 + index),
            index,
        )
        timeline.add_still(
            board,
            clip_duration,
            "impact-" + ("push", "whip-left", "whip-right", "pull")[index],
            accent=True,
        )
    if video_times:
        for index, clip_duration in enumerate(raw_durations):
            source, source_time = video_times[index % len(video_times)]
            modes = (
                "punch", "close-left", "speed", "whip", "close-right", "beat",
                "hflip", "close-high", "pull", "detail", "close-low", "beat",
            )
            mode = modes[index % len(modes)]
            cut_time = raw_start + sum(raw_durations[:index])
            # Let detected musical attacks lead the climax.  A sparse
            # periodic fallback covers weak analyses without flashing on
            # three out of every four shots and creating off-beat events.
            accent = _is_accent(cut_time, strong) or index % 3 == 0
            timeline.add_video(source, max(0, source_time - clip_duration * 0.20), clip_duration, mode, accent=accent)
            _notify(progress, "climax", 62 + round(28 * (index + 1) / max(1, len(raw_durations))))
    else:
        for index, clip_duration in enumerate(raw_durations):
            cut_time = raw_start + sum(raw_durations[:index])
            photo_modes = ("push", "whip-left", "pull", "whip-right", "pulse")
            accent = _is_accent(cut_time, strong) or index % 3 == 0
            timeline.add_still(images[index % len(images)], clip_duration, photo_modes[index % len(photo_modes)], accent=accent)
    timeline.finish()
    return {"scenes": len(timeline.clips), "drop_time": drop, "mode": "reference_edit"}


def _silhouette_board(subject_path: Path, destination: Path, progress_value: float, questions: int = 0, reveal: float = 0.0) -> Path:
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), "white")
    subject = _fit_subject(_open_rgba(subject_path), 400, 680)
    alpha = subject.getchannel("A") if "A" in subject.getbands() else Image.new("L", subject.size, 255)
    silhouette = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    silhouette.putalpha(alpha)
    x, y = (WIDTH - subject.width) // 2, (HEIGHT - subject.height) // 2
    if progress_value < 1.0:
        # Reveal the silhouette in horizontal fragments rather than fading it.
        fragment_mask = Image.new("L", subject.size, 0)
        draw_mask = ImageDraw.Draw(fragment_mask)
        bands = 9
        for band in range(max(1, round(progress_value * bands))):
            top = round(subject.height * band / bands)
            bottom = round(subject.height * (band + 0.62) / bands)
            draw_mask.rectangle((0, top, subject.width, bottom), fill=255)
        fragment_mask = Image.composite(alpha, Image.new("L", subject.size, 0), fragment_mask)
        silhouette.putalpha(fragment_mask)
    canvas.alpha_composite(silhouette, (x, y))
    if reveal > 0:
        coloured = subject.copy()
        coloured.putalpha(alpha.point(lambda value: int(value * reveal)))
        canvas.alpha_composite(coloured, (x, y))
    if questions:
        draw = ImageDraw.Draw(canvas)
        question = "?" * questions
        font = _font(64)
        box = draw.textbbox((0, 0), question, font=font, stroke_width=5)
        qx = (WIDTH - (box[2] - box[0])) // 2
        qy = int(HEIGHT * 0.49)
        draw.text((qx, qy), question, font=font, fill="#42ff70", stroke_width=5, stroke_fill="#004d16")
    return _save(canvas, destination)


def _reveal_composite(cutout: Path, source: Path, destination: Path, index: int) -> Path:
    background = _cover(Image.open(source)).filter(ImageFilter.GaussianBlur(9)).convert("RGBA")
    overlay = Image.new("RGBA", background.size, (0, 0, 0, 65))
    background.alpha_composite(overlay)
    subject = _fit_subject(_open_rgba(cutout), 430, 700)
    x = (WIDTH - subject.width) // 2 + (-22 if index % 2 else 22)
    y = (HEIGHT - subject.height) // 2
    _paste_with_rim(background, subject, (x, y))
    return _save(background, destination)


def _render_mystery_reveal(
    paths: Sequence[Path], images: list[Path], video_times: list[tuple[Path, float]], cutouts: list[Path], audio: Path,
    job_dir: Path, destination: Path, duration: float, rhythm: dict, title: str, progress: Progress | None,
) -> dict:
    onsets, strong, beat = _onsets(rhythm, duration), _strong_onsets(rhythm, duration), _beat(rhythm)
    drop = _drop_time(rhythm, duration, duration * 0.55)
    fragment_end = _snap(min(drop * 0.31, 2.4), onsets, 0.16)
    timeline = _Timeline(job_dir, audio, destination, duration, progress)
    subject = cutouts[0]
    _notify(progress, "silhouette", 10)
    fragment_count = max(4, min(9, round(fragment_end / max(beat * 0.5, 0.16))))
    for index, clip_duration in enumerate(_durations(0, fragment_end, fragment_count, onsets, 4)):
        board = _silhouette_board(subject, job_dir / f"mystery-fragment-{index:02d}.png", (index + 1) / fragment_count)
        timeline.add_still(board, clip_duration, "whip-left" if index % 2 else "hold", accent=False)

    _notify(progress, "suspense", 34)
    suspense_count = max(3, round((drop - fragment_end) / max(beat * 2, 0.70)))
    suspense_durations = _durations(fragment_end, drop, suspense_count, onsets, 8)
    for index, clip_duration in enumerate(suspense_durations):
        questions = min(3, 1 + (index * 3 // max(1, suspense_count)))
        reveal = 0.0 if index < suspense_count - 2 else (index - suspense_count + 3) * 0.28
        board = _silhouette_board(subject, job_dir / f"mystery-question-{index:02d}.png", 1.0, questions, reveal)
        timeline.add_still(board, clip_duration, "hold")

    _notify(progress, "reveal", 58)
    montage_count = max(5, min(24, round((duration - drop) / max(beat, 0.30))))
    montage_durations = _durations(drop, duration, montage_count, onsets, 5)
    for index, clip_duration in enumerate(montage_durations):
        cut_time = drop + sum(montage_durations[:index])
        accent = index == 0 or _accent_or_pattern(cut_time, strong, index, 6)
        if video_times and index % 3 != 1:
            source, source_time = video_times[(index * 5) % len(video_times)]
            timeline.add_video(source, source_time, clip_duration, "punch" if index % 2 == 0 else "whip", accent=accent)
        else:
            board = _reveal_composite(cutouts[index % len(cutouts)], images[index % len(images)], job_dir / f"mystery-montage-{index:02d}.png", index)
            timeline.add_still(board, clip_duration, "whip-right" if index % 2 else "push", accent=accent)
        _notify(progress, "montaggio", 58 + round(32 * (index + 1) / montage_count))
    timeline.finish()
    return {"scenes": len(timeline.clips), "drop_time": drop, "preset": "mystery_reveal"}


def _polygon_mask(size: tuple[int, int], points: Sequence[tuple[int, int]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def _strip_board(source_paths: Sequence[Path], cutout_paths: Sequence[Path], destination: Path, title: str, phase: int, index: int) -> Path:
    background = _cover(Image.open(source_paths[index % len(source_paths)])).filter(ImageFilter.GaussianBlur(5)).convert("RGBA")
    veil = Image.new("RGBA", background.size, (0, 0, 0, 72))
    background.alpha_composite(veil)
    title_text = (title or "SOCIAL EDIT").upper()

    if phase == 0:  # diagonal moving panel
        foreground = _cover(Image.open(source_paths[(index + 1) % len(source_paths)])).convert("RGBA")
        offset = -200 + (index % 7) * 130
        mask = _polygon_mask((WIDTH, HEIGHT), [(offset, 0), (offset + 220, 0), (offset + 610, HEIGHT), (offset + 390, HEIGHT)])
        background.paste(foreground, (0, 0), mask)
    elif phase == 1:  # card stack
        for card_index in range(1 + index % 4):
            card = _cover(Image.open(source_paths[(index + card_index) % len(source_paths)]), (162, 288)).convert("RGBA")
            card = ImageOps.expand(card, border=4, fill="white")
            x = 44 + card_index * 83
            y = 225 + (card_index % 2) * 150
            background.alpha_composite(card, (x, y))
    elif phase == 2:  # V / X split masks
        left = _cover(Image.open(source_paths[(index + 1) % len(source_paths)])).convert("RGBA")
        right = _cover(Image.open(source_paths[(index + 2) % len(source_paths)])).convert("RGBA")
        if index % 2:
            mask_a = _polygon_mask((WIDTH, HEIGHT), [(0, 0), (WIDTH // 2, HEIGHT // 2), (0, HEIGHT)])
            mask_b = _polygon_mask((WIDTH, HEIGHT), [(WIDTH, 0), (WIDTH // 2, HEIGHT // 2), (WIDTH, HEIGHT)])
        else:
            mask_a = _polygon_mask((WIDTH, HEIGHT), [(0, 0), (WIDTH, HEIGHT), (0, HEIGHT)])
            mask_b = _polygon_mask((WIDTH, HEIGHT), [(WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)])
        background.paste(left, (0, 0), mask_a)
        background.paste(right, (0, 0), mask_b)
    else:  # narrow strips
        for strip in range(5):
            panel = _cover(Image.open(source_paths[(index + strip) % len(source_paths)])).convert("RGBA")
            x0 = strip * WIDTH // 5
            mask = Image.new("L", (WIDTH, HEIGHT), 0)
            ImageDraw.Draw(mask).rectangle((x0, 0, x0 + WIDTH // 5 + 2, HEIGHT), fill=255)
            background.paste(panel, (0, 0), mask)

    # Paint the title after the panels but before the subject: this makes the
    # typography readable while the cutout convincingly crosses in front.
    draw = ImageDraw.Draw(background)
    title_font = _text_fit(draw, title_text, WIDTH - 20, 132, condensed=True)
    box = draw.textbbox((0, 0), title_text, font=title_font, stroke_width=2)
    tx = (WIDTH - (box[2] - box[0])) // 2
    draw.text((tx, 390), title_text, font=title_font, fill="white", stroke_width=2, stroke_fill="black")

    subject = _fit_subject(_open_rgba(cutout_paths[index % len(cutout_paths)]), 420, 680)
    _paste_with_rim(background, subject, ((WIDTH - subject.width) // 2, (HEIGHT - subject.height) // 2 + 55))
    return _save(background, destination)


def _render_kinetic_strips(
    paths: Sequence[Path], images: list[Path], video_times: list[tuple[Path, float]], cutouts: list[Path], audio: Path,
    job_dir: Path, destination: Path, duration: float, rhythm: dict, title: str, progress: Progress | None,
) -> dict:
    onsets, strong, beat = _onsets(rhythm, duration), _strong_onsets(rhythm, duration), _beat(rhythm)
    drop = _drop_time(rhythm, duration, duration * 0.31)
    boundaries = [0.0, drop]
    for ratio in (0.54, 0.70, 0.85):
        boundaries.append(_snap(drop + (duration - drop) * ratio, onsets, 0.14))
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))
    phases = (0, 1, 2, 3)
    timeline = _Timeline(job_dir, audio, destination, duration, progress)
    _notify(progress, "titolo cinetico", 8)
    for section_index in range(len(boundaries) - 1):
        start, end = boundaries[section_index], boundaries[section_index + 1]
        phase = phases[min(section_index, len(phases) - 1)]
        unit = beat * (0.5 if section_index == 0 else 1.0)
        count = max(2, min(18, round((end - start) / max(0.16, unit))))
        section_durations = _durations(start, end, count, onsets, 5)
        for index, clip_duration in enumerate(section_durations):
            global_index = len(timeline.clips)
            board = _strip_board(images, cutouts, job_dir / f"strips-{global_index:03d}.png", title, phase, index)
            mode = ("whip-left", "push", "whip-right", "pull")[(global_index + phase) % 4]
            cut_time = start + sum(section_durations[:index])
            timeline.add_still(board, clip_duration, mode, accent=index == 0 or _accent_or_pattern(cut_time, strong, global_index, 8))
            _notify(progress, ("pannelli diagonali", "card stack", "split V/X", "strisce")[phase], 10 + round(80 * (end / duration)))
    timeline.finish()
    return {"scenes": len(timeline.clips), "drop_time": drop, "preset": "kinetic_strips"}


def _render_beat_montage(
    paths: Sequence[Path], images: list[Path], video_times: list[tuple[Path, float]], cutouts: list[Path], audio: Path,
    job_dir: Path, destination: Path, duration: float, rhythm: dict, title: str, progress: Progress | None,
) -> dict:
    onsets, strong, beat = _onsets(rhythm, duration), _strong_onsets(rhythm, duration), _beat(rhythm)
    drop = _drop_time(rhythm, duration, _section_boundary(rhythm, "drop", duration) or min(duration * 0.45, 4.2))
    # Reference: one shot every two beats.  Movement is concentrated at the
    # beginning/end, leaving the middle readable and sharp.
    count = max(3, min(30, round(duration / max(0.36, beat * 2))))
    durations = _durations(0, duration, count, onsets, 10)
    timeline = _Timeline(job_dir, audio, destination, duration, progress)
    _notify(progress, "montaggio sul beat", 10)
    for index, clip_duration in enumerate(durations):
        cut_time = sum(durations[:index])
        accent = index > 0 and (_accent_or_pattern(cut_time, strong, index, 4) or abs(cut_time - drop) < beat * 0.35)
        if video_times:
            source, source_time = video_times[(index * 5) % len(video_times)]
            # Punch/whip consumes only the first 2-5 frames, after which the
            # source stays crisp. Alternating it makes adjacent motion coherent.
            mode = "whip" if index % 2 else "punch"
            timeline.add_video(source, max(0, source_time - 0.10), clip_duration, mode, accent)
        else:
            mode = "whip-left" if index % 2 else "whip-right"
            timeline.add_still(images[index % len(images)], clip_duration, mode, accent)
        _notify(progress, "montaggio sul beat", 10 + round(80 * (index + 1) / count))
    timeline.finish()
    return {"scenes": len(timeline.clips), "drop_time": drop, "preset": "beat_montage"}


def render_preset(
    style,
    paths,
    audio_path,
    job_dir,
    destination,
    duration,
    rhythm,
    title,
    cutouts=None,
    samples=None,
    progress=None,
) -> dict:
    """Render a reference-driven preset and return timeline metadata.

    Parameters intentionally use only basic Python/Pillow/OpenCV types so the
    function can be called from Flask workers, tests, or a future queue worker.
    """
    style = str(style).strip().lower()
    supported = {"reference_edit"}
    if style not in supported:
        raise ValueError(f"Unknown preset {style!r}; expected one of {sorted(supported)}")
    paths = [Path(path) for path in paths]
    audio_path, job_dir, destination = Path(audio_path), Path(job_dir), Path(destination)
    duration = round(max(1.0, float(duration)) * FPS) / FPS
    rhythm = dict(rhythm or {})
    title = str(title or "").strip()
    job_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

    _notify(progress, "analisi contenuti", 2)
    images, video_times = _source_assets(paths, samples, job_dir)
    if not images and not video_times:
        raise ValueError("No readable photo or video source was provided")
    if not images and video_times:
        for index, (source, source_time) in enumerate(video_times[:4]):
            images.append(_extract_frame(source, source_time, job_dir / f"engine-fallback-{index:02d}.jpg"))
    subjects = _cutout_assets(cutouts, images, job_dir)
    renderer = _render_reference_edit
    result = renderer(paths, images, video_times, subjects, audio_path, job_dir, destination, duration, rhythm, title, progress)
    _notify(progress, "completato", 100)
    return result


__all__ = ["render_preset"]
