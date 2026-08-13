import json
import unittest

import numpy as np

from edit_timeline import (
    REFERENCE_DROP_RATIO,
    REFERENCE_TAIL_RATIO,
    build_edit_timeline,
    build_sync_grid,
    choose_audio_window,
)


FPS = 30


def _seconds(frame):
    return frame / FPS


class EditTimelineTests(unittest.TestCase):
    def test_late_108_bpm_drop_uses_musical_start_and_valid_tail_window(self):
        # Real failing-export geometry: a 30 s song whose detected drop is at
        # 20.466 s.  A 24 s request cannot retain a 52.7% post-drop tail.
        rhythm = {
            "duration": 30.0,
            "drop_time": 20.4666667,
            "beat_times": [index * (60 / 108) for index in range(55)],
            "onsets": [1.0, 5.0, 10.0, 14.0, 18.0, 22.0, 25.0, 28.0],
            "onset_strengths": [0.4, 0.7, 0.5, 0.8, 0.6, 0.9, 0.5, 0.7],
            "energy_curve": [],
        }

        window = choose_audio_window(rhythm, 24.0)

        self.assertEqual(window["source_visual_drop_frame"], 614)
        self.assertEqual(window["start_frame"], 367)
        self.assertEqual(window["end_frame"], 900)
        self.assertEqual(window["duration_frames"], 533)
        self.assertEqual(window["visual_drop_frame"], 247)
        self.assertEqual(window["tail_frames"], 286)
        self.assertLessEqual(window["duration_frames"], 24 * FPS)
        self.assertGreaterEqual(window["tail_ratio"] + 1e-6, REFERENCE_TAIL_RATIO)
        self.assertTrue(window["start_on_grid"])
        musical_frames = {
            round(value * FPS)
            for value in [*rhythm["beat_times"], *rhythm["onsets"]]
        }
        self.assertLessEqual(
            min(abs(window["start_frame"] - frame) for frame in musical_frames), 2
        )
        # Snapping costs less than one beat and avoids a visibly arbitrary
        # first cut in the middle of the music grid.
        self.assertLessEqual(542 - window["duration_frames"], round(FPS * 60 / 108))

    def test_reference_cluster_recovers_measured_window_and_cues(self):
        duration_frames = 727
        beat_times = [index * (60 / 160) for index in range(66)]
        important_frames = [
            97,
            255,
            276,
            283,
            290,
            312,
            325,
            344,
            350,
            356,
            361,
            367,
        ]
        rhythm = {
            "duration": _seconds(duration_frames),
            "drop_time": _seconds(361),
            "beat_times": beat_times,
            "onsets": [_seconds(frame) for frame in important_frames],
            "onset_strengths": [0.93, 0.88, 0.9, 0.75, 0.82, 0.91, 0.78, 0.96, 0.9, 0.92, 1.0, 0.89],
            "energy_curve": [
                {"time": _seconds(340), "energy": 0.25, "flux": 0.3},
                {"time": _seconds(361), "energy": 0.92, "flux": 1.0},
            ],
        }

        result = build_edit_timeline(rhythm, _seconds(duration_frames))
        window = result["window"]
        cues = result["cues"]

        self.assertEqual(window["start_frame"], 0)
        self.assertEqual(window["end_frame"], 727)
        self.assertEqual(window["source_energy_drop_frame"], 361)
        self.assertEqual(window["source_visual_drop_frame"], 344)
        self.assertEqual(window["visual_drop_frame"], 344)
        self.assertEqual(cues["boundaries"]["intro_end_frame"], 97)
        self.assertEqual(cues["boundaries"]["roulette_end_frame"], 255)
        self.assertEqual(cues["boundaries"]["build_end_frame"], 276)
        self.assertEqual(cues["boundaries"]["word_frames"], [276, 283, 290, 312, 325, 344])
        self.assertEqual(cues["boundaries"]["impact_frames"], [344, 350, 356, 361, 367])
        self.assertEqual(cues["boundaries"]["climax_start_frame"], 367)

        # Quarter-beat roulette cuts stay native.  There is deliberately no
        # historical max-48 limiter that would stretch their durations.
        self.assertGreater(len(cues["roulette"]["cut_frames"]), 48)

        grid_frames = {point["frame"] for point in result["sync_grid"]}
        boundaries = cues["boundaries"]
        required = {
            boundaries["intro_end_frame"],
            boundaries["roulette_end_frame"],
            boundaries["build_end_frame"],
            boundaries["drop_frame"],
            boundaries["climax_start_frame"],
            *boundaries["word_frames"],
            *boundaries["impact_frames"],
        }
        self.assertTrue(required.issubset(grid_frames))
        self.assertEqual([word["text"] for word in cues["words"]], ["CAN", "YOU", "IMAGINE", "FLOATING", "WEIGHTLESS"])
        self.assertGreater(cues["impacts"][0]["accent_strength"], 0.9)

        # This is an API contract for the web app, so no dataclasses, NumPy
        # scalars or other non-JSON values may leak out.
        json.dumps(result)

    def test_grid_uses_strongest_onset_within_three_frames(self):
        rhythm = {
            "duration": 4.0,
            "drop_time": 2.0,
            "beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            # The nominal half-beat is F8.  F7 is closer, but F10 is the
            # stronger onset and is still inside the permitted three frames.
            "onsets": [_seconds(7), _seconds(10)],
            "onset_strengths": [0.25, 0.95],
        }
        window = {
            "start_frame": 0,
            "end_frame": 120,
            "duration_frames": 120,
            "source_duration_frames": 120,
        }

        grid = build_sync_grid(rhythm, window)

        snapped = next(point for point in grid if point["source_frame"] == 10)
        self.assertTrue(snapped["snapped"])
        self.assertTrue(snapped["onset"])
        self.assertEqual(snapped["accent_strength"], 0.95)
        self.assertEqual(snapped["level"], "half")

    def test_onset_snapping_does_not_erase_quarter_beat_density(self):
        rhythm = {
            "duration": 4.0,
            "drop_time": 2.0,
            "beat_times": [index * (60 / 108) for index in range(9)],
            # Put transients three frames after nominal subdivisions so two
            # neighbours would collapse without retaining the nominal grid.
            "onsets": [_seconds(frame) for frame in (3, 20, 37, 54, 70, 87, 104)],
            "onset_strengths": [0.9] * 7,
        }
        window = {
            "start_frame": 0,
            "end_frame": 120,
            "duration_frames": 120,
            "source_duration_frames": 120,
        }
        grid = build_sync_grid(rhythm, window)
        frames = sorted(point["frame"] for point in grid)
        # 108 BPM quarter-beats are roughly 4.2 frames apart. The grid may
        # contain extra onset frames, but it must not develop long holes.
        gaps = [right - left for left, right in zip(frames, frames[1:])]
        self.assertLessEqual(np.percentile(gaps, 90), 5)

    def test_energy_curve_is_a_safe_drop_fallback(self):
        rhythm = {
            "duration": 10.0,
            "beat_times": [index * 0.5 for index in range(21)],
            "onsets": [],
            "energy_curve": [
                {"time": 2.0, "energy": 0.2, "flux": 0.1},
                {"time": 5.0, "energy": 0.85, "flux": 0.9},
                {"time": 8.0, "energy": 0.8, "flux": 0.2},
            ],
        }

        window = choose_audio_window(rhythm, 8.0)

        self.assertEqual(window["source_energy_drop_frame"], 150)
        self.assertEqual(window["source_visual_drop_frame"], 150)
        self.assertTrue(all(isinstance(window[key], int) for key in ("start_frame", "end_frame", "visual_drop_frame")))

    def test_extreme_drops_and_short_tracks_keep_positive_monotonic_cues(self):
        cases = [
            (0.10, 0.001),
            (1.00, 0.983),
            (3.00, 2.949),
            (30.0, 0.20),
            (30.0, 29.50),
        ]
        for duration, drop in cases:
            with self.subTest(duration=duration, drop=drop):
                result = build_edit_timeline(
                    {
                        "duration": duration,
                        "drop_time": drop,
                        "bpm": 120,
                        "edit_bpm": 120,
                        "beat_times": [],
                        "onsets": [],
                        "onset_strengths": [],
                    },
                    24.0,
                )
                cues = result["cues"]
                total = result["window"]["duration_frames"]
                pre_drop = [cues["intro"], cues["roulette"], cues["build"], *cues["words"]]
                segments = [*pre_drop, cues["climax"]]
                self.assertTrue(
                    all(
                        0 <= item["start_frame"] < item["end_frame"] <= total
                        for item in segments
                    )
                )
                self.assertTrue(
                    all(
                        left["end_frame"] == right["start_frame"]
                        for left, right in zip(pre_drop, pre_drop[1:])
                    )
                )
                impacts = [item["frame"] for item in cues["impacts"]]
                self.assertEqual(impacts, sorted(set(impacts)))
                self.assertEqual(cues["climax"]["start_frame"], impacts[-1])
                self.assertEqual(cues["climax"]["end_frame"], total)
                self.assertTrue(cues["degraded"])


if __name__ == "__main__":
    unittest.main()
