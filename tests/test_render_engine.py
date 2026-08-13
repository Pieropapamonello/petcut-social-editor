import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inspect

from render_engine import (
    _drop_time,
    _durations,
    _finish,
    _intro_video_clip,
    _intro_asset_groups,
    _onset_aligned_weighted_durations,
    _reference_cues,
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

    def test_reference_cues_reproduce_measured_frames(self):
        cues = _reference_cues(344 / 30, 60 / 160, 727 / 30)
        self.assertEqual(cues["intro_slots"], 4)
        self.assertEqual(round(float(cues["intro_end"]) * 30), 97)
        self.assertEqual(round(float(cues["roulette_end"]) * 30), 255)
        self.assertEqual(round(float(cues["phrase_start"]) * 30), 276)
        self.assertEqual(round(float(cues["drop"]) * 30), 344)

    def test_short_build_keeps_words_readable_with_two_cards(self):
        beat = 60 / 138
        cues = _reference_cues(214 / 30, beat, 437 / 30)
        self.assertEqual(cues["intro_slots"], 2)
        word_frames = round((float(cues["drop"]) - float(cues["phrase_start"])) * 30)
        self.assertGreaterEqual(word_frames, 75)

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

    def test_intro_video_does_not_fade_from_white(self):
        with tempfile.TemporaryDirectory() as directory, patch("render_engine._run") as run:
            root = Path(directory)
            _intro_video_clip(root / "source.mp4", root / "overlay.png", root / "out.mp4", 0.0, 0.5)
            command = " ".join(str(value) for value in run.call_args.args[0])
            self.assertNotIn("fade=t=in", command)
            self.assertNotIn("color=white", command)

    def test_intro_assets_stay_grouped_by_upload_source(self):
        paths = [Path("first.mp4"), Path("second.mp4"), Path("third.jpg")]
        images = [
            Path("sample-01-source-01.jpg"),
            Path("sample-04-source-01.jpg"),
            Path("sample-02-source-02.jpg"),
            Path("original-appended.jpg"),
        ]
        cutouts = [Path("dog-b.png"), Path("dog-b-pose.png"), Path("dog-c.png")]
        groups = _intro_asset_groups(paths, images, cutouts)
        self.assertEqual([source for source, _ in groups], [1, 2])
        self.assertEqual(groups[0][1][0], (images[0], cutouts[0]))
        self.assertEqual(groups[0][1][1], (images[1], cutouts[1]))
        self.assertEqual(groups[1][1][0], (images[2], cutouts[2]))

    def test_finish_normalises_reference_colour_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch("render_engine._run") as run:
            root = Path(directory)
            clip = root / "clip.mp4"
            _finish([clip], root / "audio.wav", root / "out.mp4", 1.0, root)
            command = [str(value) for value in run.call_args.args[0]]
            self.assertIn("libx264", command)
            self.assertIn("yuv420p", command)
            self.assertIn("tv", command)
            self.assertGreaterEqual(command.count("bt709"), 3)

    def test_climax_periodic_accents_are_sparse(self):
        source = inspect.getsource(__import__("render_engine")._render_reference_edit)
        self.assertEqual(source.count("index % 3 == 0"), 2)
        self.assertNotIn("index % 4 != 3", source)


if __name__ == "__main__":
    unittest.main()
