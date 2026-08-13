"""Frame-exact music window and cue-sheet generation for the reference edit.

The renderer deliberately stays out of this module.  Its only job is to turn
the audio analysis dictionary into JSON-safe source-window, sync-grid and cue
data.  Frames are the canonical unit; seconds are included only as a
convenience for FFmpeg callers.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Sequence


FPS = 30

# Measured on Download (5): the first visual impact is F344 of 727 and the
# usable tail is the remaining 383 frames.  The latter rounds to 52.7%.
REFERENCE_DROP_RATIO = 344 / 727
REFERENCE_TAIL_RATIO = 383 / 727
REFERENCE_ACT_RATIOS = {
    "intro_end": 97 / 344,
    "roulette_end": 255 / 344,
    "build_end": 276 / 344,
}
REFERENCE_WORDS = ("CAN", "YOU", "IMAGINE", "FLOATING", "WEIGHTLESS")
REFERENCE_WORD_WEIGHTS = (7, 7, 22, 13, 19)
MIN_ACT_FRAMES = 3
MIN_WORD_FRAMES = 1
MIN_POST_DROP_FRAMES = 1
MIN_TIMELINE_FRAMES = MIN_ACT_FRAMES + MIN_WORD_FRAMES + MIN_POST_DROP_FRAMES


def _number(value: object, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _frame(value: object, fps: int) -> int:
    return int(round(_number(value) * fps))


def _seconds(frame: int, fps: int) -> float:
    return round(frame / fps, 6)


def _normalised_onsets(rhythm: dict, source_frames: int, fps: int) -> list[tuple[int, float]]:
    values = rhythm.get("onsets") or []
    strengths = rhythm.get("onset_strengths")
    if strengths is None:
        strengths = rhythm.get("strengths")
    strengths = strengths or []

    by_frame: dict[int, float] = {}
    for index, value in enumerate(values):
        frame = _frame(value, fps)
        if not 0 <= frame <= source_frames:
            continue
        strength = _number(strengths[index], 0.0) if index < len(strengths) else 0.0
        by_frame[frame] = max(by_frame.get(frame, 0.0), max(0.0, strength))
    return sorted(by_frame.items())


def _beat_period_frames(rhythm: dict, fps: int) -> int:
    beat_frames = sorted({_frame(value, fps) for value in rhythm.get("beat_times") or []})
    gaps = [right - left for left, right in zip(beat_frames, beat_frames[1:]) if right > left]
    if gaps:
        return max(4, int(round(median(gaps))))
    bpm = _number(rhythm.get("edit_bpm"), _number(rhythm.get("bpm"), 120.0))
    bpm = min(240.0, max(40.0, bpm))
    return max(4, int(round(60.0 * fps / bpm)))


def _energy_drop_frame(rhythm: dict, source_frames: int, fps: int) -> int:
    supplied = _number(rhythm.get("drop_time"), -1.0)
    if supplied >= 0.0:
        return min(source_frames - 1, max(1, _frame(supplied, fps)))

    # A useful deterministic fallback for callers that only have the compact
    # energy curve: prefer a high-flux positive energy transition.
    curve = rhythm.get("energy_curve") or []
    best: tuple[float, int] | None = None
    previous_energy = 0.0
    for point in curve:
        if not isinstance(point, dict):
            continue
        frame = _frame(point.get("time"), fps)
        if not 1 <= frame < source_frames:
            continue
        energy = _number(point.get("energy"), 0.0)
        flux = _number(point.get("flux"), 0.0)
        score = flux + max(0.0, energy - previous_energy) * 1.4
        previous_energy = energy
        if best is None or score > best[0]:
            best = (score, frame)
    return best[1] if best else min(source_frames - 1, max(1, round(source_frames * REFERENCE_DROP_RATIO)))


def _best_onset(
    onsets: Sequence[tuple[int, float]], target: int, radius: int = 3
) -> tuple[int, float] | None:
    nearby = [(frame, strength) for frame, strength in onsets if abs(frame - target) <= radius]
    if not nearby:
        return None
    # "Best" means strongest transient first, then closest and finally the
    # earliest frame.  This is more stable than snapping to the first hit.
    return max(nearby, key=lambda item: (item[1], -abs(item[0] - target), -item[0]))


def _visual_drop_frame(rhythm: dict, source_frames: int, fps: int) -> tuple[int, int]:
    """Return ``(visual_drop, energy_drop)`` in source-audio frames.

    Download (5) starts its visual impact cluster one and a half beats before
    the energy-section boundary.  We use that precursor only when an actually
    strong onset exists within three frames of the expected position.
    """

    energy_drop = _energy_drop_frame(rhythm, source_frames, fps)
    onsets = _normalised_onsets(rhythm, source_frames, fps)
    beat = _beat_period_frames(rhythm, fps)
    precursor = _best_onset(onsets, round(energy_drop - 1.5 * beat), 3)
    if precursor is not None and precursor[1] >= 0.72:
        visual_drop = precursor[0]
    else:
        at_drop = _best_onset(onsets, energy_drop, 3)
        visual_drop = at_drop[0] if at_drop is not None else energy_drop

    # Even a malformed/very short analysis must leave one positive frame for
    # intro, roulette, build and a word, plus one frame after the drop.  Real
    # analyses already satisfy this (their drop search excludes the edges),
    # so this only normalises pathological input dictionaries.
    minimum_pre = MIN_ACT_FRAMES + MIN_WORD_FRAMES
    if source_frames >= minimum_pre + MIN_POST_DROP_FRAMES:
        visual_drop = max(minimum_pre, min(source_frames - MIN_POST_DROP_FRAMES, visual_drop))
    return visual_drop, energy_drop


def _musical_source_frames(rhythm: dict, source_frames: int, fps: int) -> list[int]:
    """Return supplied (or inferred) musical boundaries in source frames."""

    supplied = {
        _frame(value, fps)
        for value in [*(rhythm.get("beat_times") or []), *(rhythm.get("onsets") or [])]
    }
    supplied = {value for value in supplied if 0 <= value <= source_frames}
    if supplied:
        return sorted(supplied)

    period = _beat_period_frames(rhythm, fps)
    anchor = _energy_drop_frame(rhythm, source_frames, fps) % period
    return list(range(anchor, source_frames + 1, period))


def choose_audio_window(rhythm: dict, requested_duration: float, fps: int = FPS) -> dict:
    """Choose the longest feasible frame-exact window around the visual drop.

    The search is intentionally length-first.  For each candidate length it
    places the drop as close as possible to 47.3%, while preserving the
    measured 52.7% post-drop tail and staying inside the source audio.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    # The audio analyser itself reports at least one second for an unreadable
    # or sub-second file.  Mirror that guarantee here so the cue sheet always
    # has enough frames for an explicit degraded four-act structure.
    source_duration = max(1.0, _number(rhythm.get("duration"), requested_duration))
    source_frames = max(fps, _frame(source_duration, fps))
    requested_frames = min(source_frames, max(2, _frame(requested_duration, fps)))
    visual_drop, energy_drop = _visual_drop_frame(rhythm, source_frames, fps)

    chosen: tuple[int, int, int] | None = None
    for length in range(requested_frames, 0, -1):
        minimum_tail = int(math.ceil(length * REFERENCE_TAIL_RATIO - 1e-9))
        maximum_local_drop = length - minimum_tail
        low = max(0, visual_drop + length - source_frames)
        high = min(visual_drop, maximum_local_drop)
        if low > high:
            continue
        target = int(round(length * REFERENCE_DROP_RATIO))
        local_drop = max(low, min(high, target))
        start = visual_drop - local_drop
        chosen = start, start + length, local_drop
        break

    if chosen is None:  # Only reachable for a pathological drop at EOF.
        end = source_frames
        start = max(0, end - requested_frames)
        local_drop = min(requested_frames - 1, max(0, visual_drop - start))
        chosen = start, end, local_drop

    start, end, local_drop = chosen

    # If the reference tail ratio would collapse an edge-drop song below the
    # smallest renderable structure, relax only that ratio.  Five frames are
    # enough for three acts, one explicit word and a positive climax.
    if end - start < MIN_TIMELINE_FRAMES and source_frames >= MIN_TIMELINE_FRAMES:
        start = max(0, visual_drop - (MIN_ACT_FRAMES + MIN_WORD_FRAMES))
        end = min(source_frames, max(visual_drop + MIN_POST_DROP_FRAMES, start + MIN_TIMELINE_FRAMES))
        if end - start < MIN_TIMELINE_FRAMES:
            start = max(0, end - MIN_TIMELINE_FRAMES)
        local_drop = visual_drop - start

    # A mathematically ideal ratio can begin between beats.  Prefer a supplied
    # beat/onset near that start, accepting at most one inferred beat of lost
    # duration, while preserving the reference tail budget and all frame
    # invariants.  If no such point exists the longest feasible window remains
    # authoritative.
    base_start, base_end = start, end
    base_length = base_end - base_start
    beat_period = _beat_period_frames(rhythm, fps)
    minimum_pre = MIN_ACT_FRAMES + MIN_WORD_FRAMES
    musical_candidates: list[tuple[int, float, int, int, int]] = []
    for candidate_start in _musical_source_frames(rhythm, source_frames, fps):
        if abs(candidate_start - base_start) > beat_period:
            continue
        candidate_end = min(source_frames, candidate_start + requested_frames)
        candidate_length = candidate_end - candidate_start
        candidate_drop = visual_drop - candidate_start
        if candidate_length < max(2, base_length - beat_period):
            continue
        if candidate_drop < minimum_pre or candidate_end - visual_drop < MIN_POST_DROP_FRAMES:
            continue
        candidate_tail = candidate_end - visual_drop
        if candidate_tail < int(math.ceil(candidate_length * REFERENCE_TAIL_RATIO - 1e-9)):
            continue
        ratio_error = abs(candidate_drop / candidate_length - REFERENCE_DROP_RATIO)
        musical_candidates.append(
            (candidate_length, -ratio_error, -abs(candidate_start - base_start), candidate_start, candidate_end)
        )
    if musical_candidates:
        _, _, _, start, end = max(musical_candidates)
        local_drop = visual_drop - start

    duration_frames = end - start
    tail_frames = duration_frames - local_drop
    musical_frames = _musical_source_frames(rhythm, source_frames, fps)
    return {
        "fps": fps,
        "source_duration_frames": source_frames,
        "source_duration": _seconds(source_frames, fps),
        "start_frame": start,
        "start": _seconds(start, fps),
        "end_frame": end,
        "end": _seconds(end, fps),
        "duration_frames": duration_frames,
        "duration": _seconds(duration_frames, fps),
        "source_visual_drop_frame": visual_drop,
        "source_visual_drop": _seconds(visual_drop, fps),
        "source_energy_drop_frame": energy_drop,
        "source_energy_drop": _seconds(energy_drop, fps),
        "visual_drop_frame": local_drop,
        "visual_drop": _seconds(local_drop, fps),
        "tail_frames": tail_frames,
        "drop_ratio": round(local_drop / duration_frames, 6),
        "tail_ratio": round(tail_frames / duration_frames, 6),
        "start_on_grid": any(abs(start - value) <= 2 for value in musical_frames),
    }


def _source_beats(rhythm: dict, start: int, end: int, fps: int) -> list[int]:
    supplied = sorted({_frame(value, fps) for value in rhythm.get("beat_times") or []})
    supplied = [value for value in supplied if value >= 0]
    period = _beat_period_frames(rhythm, fps)
    if supplied:
        beats = list(supplied)
        while beats[0] > start - period:
            beats.insert(0, beats[0] - period)
        while beats[-1] < end + period:
            beats.append(beats[-1] + period)
        return beats

    anchor = (_energy_drop_frame(rhythm, max(end, 2), fps) // period) * period
    first = anchor
    while first > start - period:
        first -= period
    beats = []
    value = first
    while value <= end + period:
        beats.append(value)
        value += period
    return beats


def build_sync_grid(rhythm: dict, window: dict, fps: int = FPS) -> list[dict]:
    """Build beat/half/quarter points, snapping each to the best ±3f onset."""

    start = int(window["start_frame"])
    end = int(window["end_frame"])
    onsets = _normalised_onsets(rhythm, int(window["source_duration_frames"]), fps)
    beats = _source_beats(rhythm, start, end, fps)
    level_rank = {"boundary": 4, "beat": 3, "half": 2, "quarter": 1}
    points: dict[int, dict] = {}

    for left, right in zip(beats, beats[1:]):
        if right <= left:
            continue
        for subdivision in range(4):
            nominal = int(round(left + (right - left) * subdivision / 4))
            if not start <= nominal <= end:
                continue
            level = "beat" if subdivision == 0 else "half" if subdivision == 2 else "quarter"
            onset = _best_onset(onsets, nominal, 3)
            source_frame = onset[0] if onset is not None else nominal
            if not start <= source_frame <= end:
                source_frame = nominal
                onset = None
            edit_frame = source_frame - start
            candidate = {
                "frame": edit_frame,
                "time": _seconds(edit_frame, fps),
                "source_frame": source_frame,
                "source_time": _seconds(source_frame, fps),
                "level": level,
                "nominal_source_frame": nominal,
                "snapped": onset is not None and onset[0] != nominal,
                "onset": onset is not None,
                "accent_strength": round(onset[1], 4) if onset is not None else 0.0,
            }
            # Keep the quantised subdivision as well as its transient snap.
            # Otherwise a strong, broad onset can pull two adjacent quarter
            # beats onto one frame and slow the roulette from ~8 to ~5 cuts/s.
            nominal_edit = nominal - start
            if nominal_edit != edit_frame and nominal_edit not in points:
                points[nominal_edit] = {
                    "frame": nominal_edit,
                    "time": _seconds(nominal_edit, fps),
                    "source_frame": nominal,
                    "source_time": _seconds(nominal, fps),
                    "level": level,
                    "nominal_source_frame": nominal,
                    "snapped": False,
                    "onset": False,
                    "accent_strength": 0.0,
                }
            previous = points.get(edit_frame)
            if previous is None:
                points[edit_frame] = candidate
            else:
                # Several neighbouring quarter-beats can legitimately snap to
                # the same broad transient.  Merging all of them made slow
                # songs lose a third of the roulette cuts (about 5/s instead
                # of the reference's 7–8/s). Keep the strongest transient at
                # its frame and retain the colliding nominal subdivision as a
                # non-onset grid point; it is still musically quantised and at
                # most three frames from that transient.
                if level_rank[level] > level_rank.get(previous["level"], 0):
                    previous["level"] = level
                    previous["nominal_source_frame"] = nominal
                previous["onset"] = previous["onset"] or candidate["onset"]
                previous["snapped"] = previous["snapped"] or candidate["snapped"]
                previous["accent_strength"] = max(previous["accent_strength"], candidate["accent_strength"])

    duration = end - start
    visual_drop = int(window.get("visual_drop_frame", 0))
    boundaries = ((0, start), (visual_drop, start + visual_drop), (duration, end))
    for edit_frame, source_frame in boundaries:
        existing = points.get(edit_frame)
        if existing is not None:
            existing["level"] = "boundary"
            continue
        onset = next(((frame, strength) for frame, strength in onsets if frame == source_frame), None)
        points[edit_frame] = {
            "frame": edit_frame,
            "time": _seconds(edit_frame, fps),
            "source_frame": source_frame,
            "source_time": _seconds(source_frame, fps),
            "level": "boundary",
            "nominal_source_frame": source_frame,
            "snapped": False,
            "onset": onset is not None,
            "accent_strength": round(onset[1], 4) if onset is not None else 0.0,
        }
    return [points[key] for key in sorted(points)]


def _pick_frame(
    frames: Sequence[int], strengths: dict[int, float], target: int, low: int, high: int
) -> int:
    candidates = [frame for frame in frames if low <= frame <= high]
    if not candidates:
        return max(low, min(high, target))
    return min(candidates, key=lambda frame: (abs(frame - target), -strengths.get(frame, 0.0), frame))


def _ordered_boundaries(
    targets: Sequence[int], frames: Sequence[int], strengths: dict[int, float], low: int, high: int
) -> list[int]:
    if not targets:
        return []
    capacity = high - low + 1
    if capacity < len(targets):
        raise ValueError("not enough frames for strictly ordered cue boundaries")
    result: list[int] = []
    for index, target in enumerate(targets):
        minimum = max(low + index, (result[-1] + 1) if result else low)
        maximum = high - (len(targets) - index - 1)
        result.append(_pick_frame(frames, strengths, target, minimum, maximum))
    return result


def _cue_point(frame: int, window_start: int, strengths: dict[int, float], fps: int) -> dict:
    return {
        "frame": frame,
        "time": _seconds(frame, fps),
        "source_frame": window_start + frame,
        "source_time": _seconds(window_start + frame, fps),
        "accent_strength": round(strengths.get(frame, 0.0), 4),
    }


def _segment(
    start: int,
    end: int,
    cuts: Iterable[int],
    window_start: int,
    strengths: dict[int, float],
    fps: int,
) -> dict:
    return {
        "start_frame": start,
        "end_frame": end,
        "start": _seconds(start, fps),
        "end": _seconds(end, fps),
        "duration_frames": end - start,
        "cut_frames": [frame for frame in cuts if start <= frame <= end],
        "start_accent_strength": round(strengths.get(start, 0.0), 4),
        "end_accent_strength": round(strengths.get(end, 0.0), 4),
        "source_start_frame": window_start + start,
        "source_end_frame": window_start + end,
    }


def build_cue_sheet(rhythm: dict, window: dict, grid: Sequence[dict], fps: int = FPS) -> dict:
    """Allocate the reference acts and every internal cut on the sync grid."""

    duration = int(window["duration_frames"])
    drop = int(window["visual_drop_frame"])
    start = int(window["start_frame"])
    grid_frames = sorted({int(point["frame"]) for point in grid})
    strengths = {int(point["frame"]): _number(point.get("accent_strength")) for point in grid}

    act_targets = [
        round(drop * REFERENCE_ACT_RATIOS["intro_end"]),
        round(drop * REFERENCE_ACT_RATIOS["roulette_end"]),
        round(drop * REFERENCE_ACT_RATIOS["build_end"]),
    ]
    intro_end, roulette_end, build_end = _ordered_boundaries(
        act_targets, grid_frames, strengths, 1, max(3, drop - 1)
    )

    word_span = drop - build_end
    word_count = min(len(REFERENCE_WORDS), max(1, word_span))
    word_names = REFERENCE_WORDS[:word_count]
    word_weights = REFERENCE_WORD_WEIGHTS[:word_count]
    weight_sum = sum(word_weights)
    cumulative = 0
    word_targets = []
    for weight in word_weights[:-1]:
        cumulative += weight
        word_targets.append(build_end + round(word_span * cumulative / weight_sum))
    word_inner = _ordered_boundaries(
        word_targets, grid_frames, strengths, build_end + 1, drop - 1
    )
    word_boundaries = [build_end, *word_inner, drop]

    beat = _beat_period_frames(rhythm, fps)
    post_span = duration - drop
    impact_count = min(4, max(0, post_span - 1))
    impact_targets = [drop + round(index * beat / 2) for index in range(1, impact_count + 1)]
    post_drop_onsets = [
        frame
        for frame in grid_frames
        if drop < frame < duration and strengths.get(frame, 0.0) > 0.0
    ]
    impact_candidates = (
        post_drop_onsets if len(post_drop_onsets) >= impact_count else grid_frames
    )
    impact_inner = (
        _ordered_boundaries(
            impact_targets, impact_candidates, strengths, drop + 1, duration - 1
        )
        if impact_targets
        else []
    )
    impact_frames = [drop, *impact_inner]
    climax_start = impact_frames[-1]

    intro_grid = [frame for frame in grid_frames if 0 <= frame <= intro_end]
    roulette_grid = [frame for frame in grid_frames if intro_end <= frame <= roulette_end]
    build_grid = [frame for frame in grid_frames if roulette_end <= frame <= build_end]
    climax_grid = [frame for frame in grid_frames if climax_start <= frame <= duration]

    words = []
    for name, left, right in zip(word_names, word_boundaries, word_boundaries[1:]):
        item = _segment(left, right, [left, right], start, strengths, fps)
        item["text"] = name
        item["accent_strength"] = round(strengths.get(left, 0.0), 4)
        words.append(item)

    return {
        "intro": _segment(0, intro_end, intro_grid, start, strengths, fps),
        "roulette": _segment(intro_end, roulette_end, roulette_grid, start, strengths, fps),
        "build": _segment(roulette_end, build_end, build_grid, start, strengths, fps),
        "words": words,
        "impacts": [_cue_point(frame, start, strengths, fps) for frame in impact_frames],
        "climax": _segment(climax_start, duration, climax_grid, start, strengths, fps),
        "boundaries": {
            "intro_end_frame": intro_end,
            "roulette_end_frame": roulette_end,
            "build_end_frame": build_end,
            "word_frames": word_boundaries,
            "drop_frame": drop,
            "impact_frames": impact_frames,
            "climax_start_frame": climax_start,
            "end_frame": duration,
        },
        "degraded": word_count < len(REFERENCE_WORDS) or impact_count < 4,
        "omitted_words": list(REFERENCE_WORDS[word_count:]),
        "impact_count": impact_count,
    }


def build_edit_timeline(rhythm: dict, requested_duration: float, fps: int = FPS) -> dict:
    """Public one-call API used by the renderer.

    The returned dictionary contains only JSON-native values.  All cue and cut
    frames use edit-local coordinates; the window and every grid point also
    expose their source-audio coordinates.
    """

    window = choose_audio_window(rhythm, requested_duration, fps)
    grid = build_sync_grid(rhythm, window, fps)
    cue_sheet = build_cue_sheet(rhythm, window, grid, fps)
    return {
        "version": "reference-sync-v1",
        "fps": fps,
        "window": window,
        "sync_grid": grid,
        "cues": cue_sheet,
    }


__all__ = [
    "FPS",
    "REFERENCE_DROP_RATIO",
    "REFERENCE_TAIL_RATIO",
    "build_cue_sheet",
    "build_edit_timeline",
    "build_sync_grid",
    "choose_audio_window",
]
