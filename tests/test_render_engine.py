import tempfile
import unittest
from pathlib import Path

from render_engine import _drop_time, _durations, render_preset


class RenderTimelineTests(unittest.TestCase):
    def test_durations_are_frame_exact_and_preserve_total(self):
        values = _durations(1.0, 4.0, 7, [1.4, 1.83, 2.27, 2.7, 3.13, 3.57], 4)
        self.assertEqual(len(values), 7)
        self.assertAlmostEqual(sum(values), 3.0, places=6)
        self.assertTrue(all(value * 30 == round(value * 30) for value in values))

    def test_drop_snaps_to_nearby_onset(self):
        rhythm = {"drop_time": 7.19, "onsets": [6.8, 7.2, 7.6]}
        self.assertEqual(_drop_time(rhythm, 15.0, 7.5), 7.2)

    def test_unknown_preset_is_rejected_before_render(self):
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
