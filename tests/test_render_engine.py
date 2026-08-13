import tempfile
import unittest
from pathlib import Path

from render_engine import (
    _drop_time,
    _durations,
    _onset_aligned_weighted_durations,
    _visual_drop_time,
    _weighted_frame_durations,
    render_preset,
)


class RenderTimelineTests(unittest.TestCase):
    def test_durations_are_frame_exact_and_preserve_total(self):
        values = _durations(1.0, 4.0, 7, [1.4, 1.83, 2.27, 2.7, 3.13, 3.57], 4)
        self.assertEqual(len(values), 7)
        self.assertAlmostEqual(sum(values), 3.0, places=6)
        self.assertTrue(all(value * 30 == round(value * 30) for value in values))

    def test_drop_snaps_to_nearby_onset(self):
        rhythm = {"drop_time": 7.19, "onsets": [6.8, 7.2, 7.6]}
        self.assertEqual(_drop_time(rhythm, 15.0, 7.5), 7.2)

    def test_visual_drop_uses_first_strong_impact_cluster(self):
        rhythm = {
            "drop_time": 12.033333,
            "edit_bpm": 160,
            "onsets": [11.466667, 11.833333, 12.033333],
            "onset_strengths": [0.81, 0.82, 0.96],
        }
        self.assertAlmostEqual(_visual_drop_time(rhythm, 24.0, 11.5), 11.466667, places=5)

    def test_weighted_cue_sheet_preserves_exact_frames(self):
        values = _weighted_frame_durations(8.5, 11.466667, [21, 7, 7, 22, 13, 19], minimum=3)
        self.assertEqual([round(value * 30) for value in values], [21, 7, 7, 22, 13, 19])

    def test_weighted_climax_snaps_to_onsets_without_drift(self):
        values = _onset_aligned_weighted_durations(12.2, 15.0, [6, 11, 12, 7], [12.67, 13.57, 14.47], minimum=4)
        boundaries = [round(12.2 * 30)]
        for value in values:
            boundaries.append(boundaries[-1] + round(value * 30))
        self.assertEqual(boundaries[-1], 450)
        self.assertTrue(all(value >= 4 / 30 for value in values))
        self.assertIn(round(12.67 * 30), boundaries)

    def test_unknown_mode_is_rejected_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                render_preset(
                    "unknown",
                    [],
                    root / "audio.wav",
                    root,
                    root / "output.mp4",
                    5,
                    {},
                    "",
                )


if __name__ == "__main__":
    unittest.main()
