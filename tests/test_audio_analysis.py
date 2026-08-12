import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from audio_analysis import analyze_audio, recommendation


SAMPLE_RATE = 16_000


def synthetic_track(duration, bpm, drop=None, alternating=False):
    count = int(round(duration * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    audio = np.zeros(count, dtype=np.float32)
    period = 60.0 / bpm
    beat_times = np.arange(0.5, duration, period)
    click_length = int(0.055 * SAMPLE_RATE)
    click_time = np.arange(click_length, dtype=np.float32) / SAMPLE_RATE
    click = np.sin(2 * np.pi * 1250 * click_time) * np.exp(-click_time * 55)
    for index, beat_time in enumerate(beat_times):
        start = int(round(beat_time * SAMPLE_RATE))
        amplitude = 0.23
        if alternating and index % 2:
            amplitude *= 0.45
        if drop is not None and beat_time >= drop:
            amplitude *= 2.8
        end = min(count, start + click_length)
        audio[start:end] += amplitude * click[: end - start]

    # The low-level bed gives RMS analysis a clear section transition without
    # hiding the individual attacks from spectral-flux peak picking.
    bed_amplitude = np.full(count, 0.018, dtype=np.float32)
    if drop is not None:
        bed_amplitude[int(drop * SAMPLE_RATE) :] = 0.16
    audio += bed_amplitude * (
        np.sin(2 * np.pi * 110 * time) + 0.45 * np.sin(2 * np.pi * 220 * time)
    )
    return np.clip(audio, -0.95, 0.95)


def write_wav(path, samples):
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())


class AudioAnalysisTests(unittest.TestCase):
    def analyse(self, samples, duration):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.wav"
            write_wav(path, samples)
            return analyze_audio(path, duration)

    def test_detects_tempo_drop_and_aligned_grid(self):
        duration = 16.0
        rhythm = self.analyse(synthetic_track(duration, 120.0, drop=8.0), duration)

        expected_keys = {
            "bpm",
            "edit_bpm",
            "onsets",
            "onset_strengths",
            "beat_times",
            "drop_time",
            "duration",
            "sections",
            "energy_curve",
        }
        self.assertEqual(expected_keys, set(rhythm))
        self.assertAlmostEqual(rhythm["edit_bpm"], 120.0, delta=7.0)
        self.assertAlmostEqual(rhythm["drop_time"], 8.0, delta=0.45)
        self.assertEqual(len(rhythm["onsets"]), len(rhythm["onset_strengths"]))
        self.assertGreater(len(rhythm["onsets"]), 20)
        self.assertTrue(any(section["name"] == "climax" for section in rhythm["sections"]))
        self.assertGreater(len(rhythm["energy_curve"]), 20)

        expected_beats = np.arange(0.5, duration, 0.5)
        distances = [min(abs(beat - actual) for actual in rhythm["beat_times"]) for beat in expected_beats]
        self.assertLess(float(np.median(distances)), 0.05)
        for timestamp in rhythm["beat_times"] + rhythm["onsets"] + [rhythm["drop_time"]]:
            self.assertAlmostEqual(timestamp * 30, round(timestamp * 30), places=4)

    def test_half_tempo_is_normalised_for_social_editing(self):
        duration = 18.0
        rhythm = self.analyse(
            synthetic_track(duration, 80.0, drop=9.0, alternating=True), duration
        )
        self.assertAlmostEqual(rhythm["bpm"], 80.0, delta=8.0)
        self.assertAlmostEqual(rhythm["edit_bpm"], 160.0, delta=12.0)
        self.assertAlmostEqual(rhythm["drop_time"], 9.0, delta=0.5)

    def test_silence_has_deterministic_fallback(self):
        duration = 10.0
        first = self.analyse(np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32), duration)
        second = self.analyse(np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32), duration)
        self.assertEqual(first, second)
        self.assertEqual(first["bpm"], 120.0)
        self.assertEqual(first["edit_bpm"], 120.0)
        self.assertEqual(first["onsets"], [])
        self.assertEqual(first["drop_time"], 5.0)

    def test_recommendations_cover_every_supported_style(self):
        rhythm = {
            "edit_bpm": 140.0,
            "drop_time": 7.2,
            "duration": 15.0,
        }
        for style in (
            "animal_roulette",
            "mystery_reveal",
            "kinetic_strips",
            "beat_montage",
        ):
            with self.subTest(style=style):
                result = recommendation(style, 15.0, rhythm, media_count=1)
                self.assertEqual(
                    {"ideal_media", "min_media", "visual_cuts", "phases", "note"},
                    set(result),
                )
                self.assertEqual(result["min_media"], 1)
                self.assertGreaterEqual(result["ideal_media"], 4)
                self.assertGreater(result["visual_cuts"], 5)
                self.assertTrue(any(phase["name"] == "drop" for phase in result["phases"]))
                self.assertIn("solo contenuto", result["note"])


if __name__ == "__main__":
    unittest.main()
