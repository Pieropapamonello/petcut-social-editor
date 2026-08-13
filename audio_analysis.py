"""Lightweight, deterministic rhythm analysis for PetCut.

The module intentionally depends only on NumPy and tools already present in the
application image.  Audio is decoded to mono at a modest sample rate and the
FFT is evaluated in small batches, so even a long upload cannot create a large
spectrogram in memory.
"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np


ANALYSIS_RATE = 16_000
FRAME_RATE = 30
MAX_ANALYSIS_SECONDS = 180.0
_EPSILON = 1e-9


def _finite_float(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _quantize_time(value: float, duration: float | None = None) -> float:
    value = max(0.0, _finite_float(value, 0.0))
    if duration is not None:
        value = min(value, max(0.0, duration))
    return round(round(value * FRAME_RATE) / FRAME_RATE, 6)


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) == 0 or width <= 1:
        return values.astype(np.float32, copy=True)
    width = min(int(width), len(values))
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    result = np.convolve(padded, np.ones(width, dtype=np.float32) / width, mode="valid")
    return result.astype(np.float32, copy=False)


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        packed = packed[: len(packed) // 3 * 3].reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) < 2:
        return samples.astype(np.float32, copy=False)
    target_length = max(1, int(round(len(samples) * target_rate / source_rate)))
    source_positions = np.arange(len(samples), dtype=np.float64)
    target_positions = np.arange(target_length, dtype=np.float64) * source_rate / target_rate
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _decode_wav(path: Path, limit: float) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        source_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = min(source.getnframes(), int(math.ceil(limit * source_rate)))
        raw = source.readframes(frame_count)
    samples = _pcm_to_float(raw, sample_width)
    samples = samples[: len(samples) // channels * channels]
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return _resample(samples, source_rate, ANALYSIS_RATE)


def _decode_audio(path: Path, requested_duration: float) -> np.ndarray:
    limit = requested_duration if requested_duration > 0 else MAX_ANALYSIS_SECONDS
    limit = min(limit, MAX_ANALYSIS_SECONDS)
    if path.suffix.lower() in {".wav", ".wave"}:
        try:
            return _decode_wav(path, limit)
        except (EOFError, ValueError, wave.Error):
            pass

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-t",
        f"{limit:.3f}",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True, timeout=45)
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=False)


def _normalise_audio(samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return samples
    # ``np.frombuffer`` over ffmpeg's stdout is read-only.  Keep the WAV path
    # zero-copy, but make a writable array before in-place sanitation when the
    # decoder returned a buffer view.
    samples = samples.astype(np.float32, copy=False)
    if not samples.flags.writeable:
        samples = samples.copy()
    samples = np.nan_to_num(samples, copy=False)
    samples = samples - np.mean(samples, dtype=np.float64)
    scale = float(np.percentile(np.abs(samples), 99.5))
    if scale > _EPSILON:
        samples = np.clip(samples / scale, -1.5, 1.5)
    return samples.astype(np.float32, copy=False)


def _features(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return frame times, normalised energy, onset envelope and feature rate."""
    fft_size = 1024
    hop = 256
    if len(samples) < fft_size:
        samples = np.pad(samples, (0, fft_size - len(samples)))
    frame_count = 1 + (len(samples) - fft_size) // hop
    frame_times = (np.arange(frame_count, dtype=np.float32) * hop + fft_size / 2) / ANALYSIS_RATE
    rms = np.empty(frame_count, dtype=np.float32)
    flux = np.zeros(frame_count, dtype=np.float32)
    window = np.hanning(fft_size).astype(np.float32)
    previous_spectrum: np.ndarray | None = None

    # A batch is about one MiB.  We never materialise the full spectrogram.
    offsets = np.arange(fft_size)
    for first in range(0, frame_count, 256):
        last = min(frame_count, first + 256)
        starts = np.arange(first, last) * hop
        block = samples[starts[:, None] + offsets[None, :]]
        rms[first:last] = np.sqrt(np.mean(block * block, axis=1) + _EPSILON)
        spectrum = np.log1p(np.abs(np.fft.rfft(block * window, axis=1)) * 12.0)
        if previous_spectrum is not None:
            flux[first] = np.maximum(spectrum[0] - previous_spectrum, 0.0).mean()
        if len(spectrum) > 1:
            flux[first + 1 : last] = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).mean(axis=1)
        previous_spectrum = spectrum[-1]

    feature_rate = ANALYSIS_RATE / hop
    energy_db = 20.0 * np.log10(rms + 1e-7)
    low, high = np.percentile(energy_db, [10, 90])
    if high - low < 1.0:
        energy = np.zeros_like(rms)
    else:
        energy = np.clip((energy_db - low) / (high - low), 0.0, 1.0)
    energy = _moving_average(energy, max(1, round(feature_rate * 0.18)))

    flux = _moving_average(flux, 3)
    flux_floor = float(np.median(flux))
    flux_high = float(np.percentile(flux, 97))
    flux_normalised = np.clip((flux - flux_floor) / (flux_high - flux_floor + _EPSILON), 0.0, 2.0)
    energy_rise = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    rise_high = float(np.percentile(energy_rise, 97))
    if rise_high > _EPSILON:
        energy_rise = np.clip(energy_rise / rise_high, 0.0, 2.0)
    onset_envelope = _moving_average(0.78 * flux_normalised + 0.22 * energy_rise, 3)
    return frame_times, energy, onset_envelope, feature_rate


def _pick_onsets(
    frame_times: np.ndarray, onset_envelope: np.ndarray, feature_rate: float, duration: float
) -> tuple[list[float], list[float], np.ndarray]:
    if len(onset_envelope) < 5 or float(np.max(onset_envelope)) < 0.04:
        return [], [], np.zeros_like(onset_envelope)

    local_width = max(5, round(feature_rate * 0.70))
    baseline = _moving_average(onset_envelope, local_width)
    deviation = _moving_average(np.abs(onset_envelope - baseline), local_width)
    threshold = baseline + 0.55 * deviation + 0.035
    local_maximum = np.ones(len(onset_envelope), dtype=bool)
    for offset in (1, 2):
        local_maximum[offset:] &= onset_envelope[offset:] >= onset_envelope[:-offset]
        local_maximum[:-offset] &= onset_envelope[:-offset] > onset_envelope[offset:]
    candidates = np.flatnonzero(local_maximum & (onset_envelope >= threshold) & (onset_envelope >= 0.055))

    # Non-maximum suppression keeps fast half-beat edits but rejects a single
    # transient being reported by several adjacent FFT frames.
    minimum_gap = max(1, round(feature_rate * 0.115))
    selected: list[int] = []
    for index in candidates[np.argsort(onset_envelope[candidates])[::-1]]:
        if all(abs(int(index) - previous) >= minimum_gap for previous in selected):
            selected.append(int(index))
    selected.sort()

    if not selected:
        return [], [], onset_envelope
    raw_strengths = onset_envelope[selected]
    strength_scale = max(float(np.percentile(raw_strengths, 95)), _EPSILON)
    strengths = np.clip(raw_strengths / strength_scale, 0.0, 1.0)

    # Quantisation can merge two peaks.  Preserve the stronger one so the time
    # and strength arrays always remain parallel.
    quantised: dict[float, float] = {}
    for index, strength in zip(selected, strengths):
        timestamp = _quantize_time(float(frame_times[index]), duration)
        quantised[timestamp] = max(quantised.get(timestamp, 0.0), float(strength))
    onsets = sorted(quantised)
    onset_strengths = [round(quantised[timestamp], 4) for timestamp in onsets]
    return onsets, onset_strengths, onset_envelope


def _lag_correlation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values) - 2:
        return 0.0
    first = values[:-lag]
    second = values[lag:]
    denominator = math.sqrt(float(np.dot(first, first)) * float(np.dot(second, second)))
    if denominator <= _EPSILON:
        return 0.0
    return max(0.0, float(np.dot(first, second)) / denominator)


def _estimate_tempo(
    onset_envelope: np.ndarray,
    feature_rate: float,
    onsets: list[float],
    strengths: list[float],
) -> tuple[float, float]:
    if len(onsets) < 3 or float(np.max(onset_envelope, initial=0.0)) < 0.04:
        return 120.0, 120.0

    centred = onset_envelope.astype(np.float64) - float(np.mean(onset_envelope))
    gaps = np.diff(np.asarray(onsets, dtype=np.float64))
    gap_weights = np.minimum(np.asarray(strengths[:-1]), np.asarray(strengths[1:])) if len(strengths) > 1 else np.array([])
    valid = (gaps >= 0.18) & (gaps <= 1.50)
    gaps = gaps[valid]
    gap_weights = gap_weights[valid]

    best_bpm = 120.0
    best_score = -1.0
    for bpm in np.arange(55.0, 180.01, 0.5):
        period = 60.0 / bpm
        lag = max(1, int(round(period * feature_rate)))
        correlation = _lag_correlation(centred, lag)
        harmonic = _lag_correlation(centred, lag * 2)
        subdivision = _lag_correlation(centred, max(1, lag // 2))

        interval_score = 0.0
        if len(gaps):
            ratio = gaps / period
            # Real songs often expose half-beats or skip a beat.  All four are
            # useful evidence, with the full beat receiving the highest weight.
            possibilities = np.asarray([0.5, 1.0, 2.0, 3.0])
            preference = np.asarray([0.72, 1.0, 0.82, 0.60])
            distance = np.abs(ratio[:, None] - possibilities[None, :])
            match = np.exp(-0.5 * (distance / 0.09) ** 2) * preference[None, :]
            values = np.max(match, axis=1)
            interval_score = float(np.average(values, weights=gap_weights + 0.05))

        # A broad prior only resolves octave ties; it cannot overpower strong
        # rhythmic evidence from the track.
        prior = math.exp(-0.5 * (math.log2(bpm / 110.0) / 0.72) ** 2)
        score = 0.57 * correlation + 0.13 * harmonic + 0.05 * subdivision + 0.21 * interval_score + 0.04 * prior
        if score > best_score:
            best_bpm, best_score = float(bpm), score

    # Report a listener-facing musical tempo in the conventional range while
    # retaining the faster pulse for editing.  A 160 BPM drum pattern is often
    # labelled 80 BPM musically, but social cuts still use its 160 BPM grid.
    edit_bpm = best_bpm
    bpm = best_bpm / 2.0 if best_bpm > 150.0 else best_bpm
    while edit_bpm < 108.0:
        edit_bpm *= 2.0
    while edit_bpm > 190.0:
        edit_bpm /= 2.0
    return round(bpm, 1), round(edit_bpm, 1)


def _aligned_beats(
    duration: float, edit_bpm: float, onsets: list[float], strengths: list[float]
) -> list[float]:
    period = 60.0 / max(1.0, edit_bpm)
    if not onsets:
        phase = 0.0
    else:
        onset_array = np.asarray(onsets, dtype=np.float64)
        strength_array = np.asarray(strengths, dtype=np.float64)
        candidates = np.unique(np.r_[np.mod(onset_array, period), np.linspace(0.0, period, 90, endpoint=False)])
        best_phase, best_score = 0.0, -1.0
        sigma = min(0.075, period * 0.18)
        for phase_candidate in candidates:
            distances = np.abs((onset_array - phase_candidate + period / 2) % period - period / 2)
            score = float(np.sum((0.25 + strength_array) * np.exp(-0.5 * (distances / sigma) ** 2)))
            if score > best_score:
                best_phase, best_score = float(phase_candidate), score
        phase = best_phase

    first = phase
    while first - period >= 0.0:
        first -= period
    while first < -1e-6:
        first += period
    raw_beats = np.arange(first, duration + period * 0.25, period)
    beats = sorted({_quantize_time(float(value), duration) for value in raw_beats if -1e-6 <= value <= duration + 1e-6})
    return beats


def _drop_time(
    frame_times: np.ndarray,
    energy: np.ndarray,
    onset_envelope: np.ndarray,
    onsets: list[float],
    strengths: list[float],
    feature_rate: float,
    duration: float,
) -> float:
    fallback = _quantize_time(duration * 0.5, duration)
    if duration < 3.0 or len(energy) < 8:
        return fallback

    smooth_energy = _moving_average(energy, max(3, round(feature_rate * 0.45)))
    smooth_flux = _moving_average(onset_envelope, max(1, round(feature_rate * 0.10)))
    start_time = max(1.0, duration * 0.14)
    end_time = min(duration - 0.9, duration * 0.88)
    candidates = np.flatnonzero((frame_times >= start_time) & (frame_times <= end_time))
    if not len(candidates):
        return fallback

    before_inner = max(1, round(feature_rate * 0.12))
    before_outer = max(before_inner + 1, round(feature_rate * 0.90))
    after_inner = max(1, round(feature_rate * 0.06))
    after_outer = max(after_inner + 1, round(feature_rate * 0.95))
    sustained_outer = max(after_outer + 1, round(feature_rate * 1.65))
    scores: list[tuple[float, float, float]] = []
    for index in candidates:
        if index < before_outer or index + after_outer >= len(smooth_energy):
            continue
        before = float(np.mean(smooth_energy[index - before_outer : index - before_inner]))
        after = float(np.mean(smooth_energy[index + after_inner : index + after_outer]))
        sustained = float(np.mean(smooth_energy[index + after_inner : min(len(smooth_energy), index + sustained_outer)]))
        flux_peak = float(np.max(smooth_flux[max(0, index - before_inner) : min(len(smooth_flux), index + after_inner + 1)]))
        jump = after - before
        score = 1.25 * jump + 0.35 * (sustained - before) + 0.22 * min(flux_peak, 1.5)
        scores.append((score, jump, float(frame_times[index])))
    if not scores:
        return fallback

    score, jump, timestamp = max(scores, key=lambda item: item[0])
    if jump < 0.12 or score < 0.28:
        return fallback

    # A visual transition belongs on the transient at the start of the lift,
    # not a few analysis frames into the RMS slope.
    if onsets:
        strong = [time for time, strength in zip(onsets, strengths) if strength >= 0.35]
        nearby = [time for time in strong if abs(time - timestamp) <= 0.24]
        if nearby:
            timestamp = min(nearby, key=lambda time: abs(time - timestamp))
    return _quantize_time(timestamp, duration)


def _segment_energy(start: float, end: float, times: np.ndarray, energy: np.ndarray) -> float:
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return float(np.mean(energy)) if len(energy) else 0.0
    return float(np.mean(energy[mask]))


def _sections(
    duration: float, drop: float, frame_times: np.ndarray, energy: np.ndarray, feature_rate: float
) -> list[dict]:
    if duration <= 0.0:
        return []
    if duration < 3.0:
        return [{"name": "body", "start": 0.0, "end": duration, "energy": round(float(np.mean(energy)), 3)}]

    intro_end = _quantize_time(min(max(1.0, drop * 0.25), max(1.0, drop - 1.0)), duration)
    boundaries: list[tuple[str, float, float]] = []
    if intro_end > 0.0:
        boundaries.append(("intro", 0.0, intro_end))

    # Split out a real low-energy break only when the final pre-drop window is
    # materially quieter than the preceding build.
    break_start = drop
    if drop - intro_end >= 2.0:
        build_energy = _segment_energy(intro_end, max(intro_end, drop - 1.2), frame_times, energy)
        pre_drop_energy = _segment_energy(max(intro_end, drop - 1.2), drop, frame_times, energy)
        if pre_drop_energy + 0.10 < build_energy:
            break_start = _quantize_time(max(intro_end, drop - min(2.2, (drop - intro_end) * 0.42)), duration)

    if break_start > intro_end:
        boundaries.append(("build", intro_end, break_start))
    if drop > break_start:
        boundaries.append(("break", break_start, drop))
    if duration > drop:
        boundaries.append(("climax", drop, duration))

    if not boundaries:
        boundaries = [("body", 0.0, duration)]
    sections = []
    for name, start, end in boundaries:
        if end - start < 1.0 / FRAME_RATE:
            continue
        sections.append(
            {
                "name": name,
                "start": _quantize_time(start, duration),
                "end": _quantize_time(end, duration),
                "energy": round(_segment_energy(start, end, frame_times, energy), 3),
            }
        )
    return sections


def _compact_energy_curve(
    frame_times: np.ndarray, energy: np.ndarray, onset_envelope: np.ndarray, feature_rate: float, duration: float
) -> list[dict]:
    if not len(frame_times):
        return []
    target_points = min(240, max(12, int(math.ceil(duration * 4))))
    step = max(1, int(math.ceil(len(frame_times) / target_points)))
    curve = []
    for first in range(0, len(frame_times), step):
        last = min(len(frame_times), first + step)
        curve.append(
            {
                "time": _quantize_time(float(np.mean(frame_times[first:last])), duration),
                "energy": round(float(np.mean(energy[first:last])), 3),
                "flux": round(float(min(1.0, np.max(onset_envelope[first:last]))), 3),
            }
        )
    return curve


def _fallback(duration: float) -> dict:
    duration = _quantize_time(max(duration, 1.0))
    drop = _quantize_time(duration * 0.5, duration)
    beats = [_quantize_time(value, duration) for value in np.arange(0.0, duration + 0.001, 0.5)]
    sections = [
        {"name": "intro", "start": 0.0, "end": drop, "energy": 0.0},
        {"name": "climax", "start": drop, "end": duration, "energy": 0.0},
    ]
    return {
        "bpm": 120.0,
        "edit_bpm": 120.0,
        "onsets": [],
        "onset_strengths": [],
        "beat_times": sorted(set(beats)),
        "drop_time": drop,
        "duration": duration,
        "sections": sections,
        "energy_curve": [],
    }


def analyze_audio(path: Path, requested_duration: float) -> dict:
    """Analyse music for a short-form edit.

    All returned timestamps are snapped to a 30 fps frame.  ``bpm`` is the
    likely musical tempo, while ``edit_bpm`` is octave-normalised to the
    108--190 BPM range used for social-video cuts.
    """
    requested_duration = max(0.0, _finite_float(requested_duration, 0.0))
    fallback_duration = min(requested_duration or 15.0, MAX_ANALYSIS_SECONDS)
    try:
        samples = _normalise_audio(_decode_audio(Path(path), requested_duration))
        actual_duration = min(len(samples) / ANALYSIS_RATE, requested_duration or MAX_ANALYSIS_SECONDS)
        duration = _quantize_time(actual_duration)
        if len(samples) < ANALYSIS_RATE or float(np.max(np.abs(samples), initial=0.0)) < 1e-5:
            return _fallback(duration or fallback_duration)

        frame_times, energy, onset_envelope, feature_rate = _features(samples)
        onsets, onset_strengths, onset_envelope = _pick_onsets(
            frame_times, onset_envelope, feature_rate, duration
        )
        bpm, edit_bpm = _estimate_tempo(onset_envelope, feature_rate, onsets, onset_strengths)
        beat_times = _aligned_beats(duration, edit_bpm, onsets, onset_strengths)
        drop = _drop_time(
            frame_times,
            energy,
            onset_envelope,
            onsets,
            onset_strengths,
            feature_rate,
            duration,
        )
        return {
            "bpm": bpm,
            "edit_bpm": edit_bpm,
            "onsets": onsets,
            "onset_strengths": onset_strengths,
            "beat_times": beat_times,
            "drop_time": drop,
            "duration": duration,
            "sections": _sections(duration, drop, frame_times, energy, feature_rate),
            "energy_curve": _compact_energy_curve(
                frame_times, energy, onset_envelope, feature_rate, duration
            ),
        }
    except (OSError, subprocess.SubprocessError, ValueError, wave.Error):
        return _fallback(fallback_duration)


REFERENCE_STYLE = "reference_edit"


def recommendation(
    style: str, duration: float, rhythm: dict, media_count: int = 0
) -> dict:
    """Describe the single four-act edit derived from the supplied reference."""
    if style != REFERENCE_STYLE:
        style = REFERENCE_STYLE
    duration = max(1.0, _finite_float(duration, _finite_float(rhythm.get("duration"), 15.0)))
    duration = _quantize_time(duration)
    edit_bpm = min(190.0, max(108.0, _finite_float(rhythm.get("edit_bpm"), 120.0)))
    beat = 60.0 / edit_bpm
    drop = _finite_float(rhythm.get("drop_time"), duration * 0.5)
    minimum_drop = duration * (0.50 if duration < 8 else 0.40)
    drop = min(max(drop, minimum_drop), max(minimum_drop, duration - 0.8))

    # Normalised from Download (5): 3.23 / 8.50 / 11.47 seconds before
    # the drop, followed by a 12.76-second full-frame climax.
    intro_end = drop * (3.23 / 11.47)
    roulette_end = drop * (8.50 / 11.47)
    phrase_build_end = drop * (9.20 / 11.47)
    intro_slots = 4 if duration >= 8 else 2
    slot_frames = max(4, round(intro_end * 30 / intro_slots))
    cut_frames = slot_frames - max(2, min(slot_frames - 2, round(slot_frames * 0.55)))
    poses_per_slot = max(1, min(4, round(cut_frames / 3)))
    intro_cuts = intro_slots * (1 + poses_per_slot)
    roulette_frames = max(1, round((roulette_end - intro_end) * 30))
    roulette_cuts = max(8, min(48, round(roulette_frames / 3.35), roulette_frames // 2))
    phrase_build_cuts = max(2, min(5, round((phrase_build_end - roulette_end) / max(0.14, beat * 0.50))))
    phrase_cuts = 5
    impact_cuts = 4 if duration - drop >= 27 / 30 else max(0, int((duration - drop) * 30 // 6))
    raw_seconds = max(0.0, duration - drop - 23 / 30)
    climax_cuts = impact_cuts + (max(1, min(60, round(37 * raw_seconds / (24.233 - 12.234)))) if raw_seconds >= 4 / 30 else 0)
    phases = [
        {"name": "intro", "start": 0.0, "end": _quantize_time(intro_end, duration), "cuts": intro_cuts, "effect": "clip con nome → cutout nero"},
        {"name": "roulette", "start": _quantize_time(intro_end, duration), "end": _quantize_time(roulette_end, duration), "cuts": roulette_cuts, "effect": "pose alternate ogni mezzo beat"},
        {"name": "frase", "start": _quantize_time(roulette_end, duration), "end": _quantize_time(drop, duration), "cuts": phrase_build_cuts + phrase_cuts, "effect": "CAN YOU IMAGINE FLOATING WEIGHTLESS"},
        {"name": "climax", "start": _quantize_time(drop, duration), "end": duration, "cuts": climax_cuts, "effect": "hard cut, whip, flash e zoom blur"},
    ]
    visual_cuts = intro_cuts + roulette_cuts + phrase_build_cuts + phrase_cuts + climax_cuts
    ideal_media = max(8, min(14, int(math.ceil(visual_cuts / 4.2))))
    minimum_media = 1
    media_count = max(0, int(_finite_float(media_count, 0.0)))

    if media_count <= 1:
        note = f"Funziona anche con un solo contenuto; per replicare la varietà del riferimento sono ideali {ideal_media} foto o clip con pose diverse."
    elif media_count < ideal_media:
        note = (
            f"Il montaggio funziona con {media_count} contenuti, ma {ideal_media} permettono "
            "più varietà nel climax senza ripetere le stesse inquadrature."
        )
    else:
        note = "Materiale sufficiente per mantenere pose diverse nella roulette e nel climax."

    return {
        "ideal_media": ideal_media,
        "min_media": minimum_media,
        "visual_cuts": visual_cuts,
        "phases": phases,
        "note": note,
    }


__all__ = ["analyze_audio", "recommendation"]
