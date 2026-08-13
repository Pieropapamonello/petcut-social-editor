import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as app_module


app = app_module.app


def click_track(duration: float = 6.0, bpm: float = 120.0) -> io.BytesIO:
    rate = 16_000
    samples = np.zeros(round(duration * rate), dtype=np.float32)
    pulse_length = round(0.04 * rate)
    pulse_time = np.arange(pulse_length, dtype=np.float32) / rate
    pulse = 0.65 * np.sin(2 * np.pi * 900 * pulse_time) * np.exp(-pulse_time * 60)
    for timestamp in np.arange(0.4, duration, 60.0 / bpm):
        start = round(timestamp * rate)
        end = min(len(samples), start + pulse_length)
        samples[start:end] += pulse[: end - start]

    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    target.seek(0)
    return target


class AppContractTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_home_and_health_expose_single_reference_mode(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Il montaggio", page.data)
        self.assertIn(b"Roulette", page.data)
        self.assertNotIn(b'name="style"', page.data)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        body = health.get_json()
        self.assertEqual(body["profile"], "reference-match-v5")
        self.assertEqual(body["output"], "576x1024")
        self.assertEqual(body["mode"], "reference_edit")
        self.assertNotIn("presets", body)

    def test_audio_analysis_contract(self):
        response = self.client.post(
            "/api/analyze-audio",
            data={
                "audio": (click_track(), "song.wav"),
                "duration": "6",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreater(body["bpm"], 0)
        self.assertGreater(body["edit_bpm"], 0)
        self.assertAlmostEqual(body["duration"], 6.0, delta=0.1)
        self.assertGreater(body["visual_cuts"], 5)
        self.assertGreaterEqual(body["recommended_content"], 1)
        self.assertTrue(body["sections"])

    def test_api_validation_is_json(self):
        bad_duration = self.client.post(
            "/api/analyze-audio",
            data={"audio": (click_track(), "song.wav"), "duration": "non-un-numero"},
            content_type="multipart/form-data",
        )
        self.assertEqual(bad_duration.status_code, 400)
        self.assertEqual(bad_duration.content_type, "application/json")
        self.assertIn("Durata", bad_duration.get_json()["error"])

        missing_job = self.client.get("/api/render/does-not-exist")
        self.assertEqual(missing_job.status_code, 404)
        self.assertEqual(missing_job.content_type, "application/json")

    def test_complete_job_can_be_recovered_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            job_dir = output_root / "jobs" / "stored"
            job_dir.mkdir(parents=True)
            (job_dir / "petcut-social-edit.mp4").write_bytes(b"video")
            (job_dir / "job.json").write_text(
                json.dumps({"job_id": "persisted", "status": "complete", "progress": 100}),
                encoding="utf-8",
            )
            app_module.JOBS.pop("persisted", None)
            with patch.object(app_module, "OUTPUT_DIR", output_root):
                response = self.client.get("/api/render/persisted")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "complete")
            self.assertIn("persisted", app_module.JOBS)

    def test_uploaded_photo_respects_exif_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "portrait.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (120, 80), "orange").save(source, exif=exif)
            frame = app_module.source_frame(source, 0, 1)
            self.assertEqual(frame.size, (80, 120))

    def test_cutout_quality_rejects_background_rectangles(self):
        rectangle = Image.new("RGBA", (160, 220), (120, 90, 70, 255))
        subject = Image.new("RGBA", (160, 220), (0, 0, 0, 0))
        yy, xx = np.mgrid[:220, :160]
        ellipse = (((xx - 80) / 54) ** 2 + ((yy - 110) / 92) ** 2) <= 1
        subject.putalpha(Image.fromarray((ellipse * 255).astype(np.uint8), "L"))
        subject_score = app_module.cutout_quality(subject)
        rectangle_score = app_module.cutout_quality(rectangle)
        self.assertGreater(subject_score, 0.25)
        self.assertGreater(subject_score, rectangle_score + 0.20)
        self.assertLess(rectangle_score, 0.20)

    def test_cutout_quality_rejects_perforated_upper_body(self):
        clean = Image.new("RGBA", (180, 260), (70, 65, 60, 0))
        yy, xx = np.mgrid[:260, :180]
        silhouette = (((xx - 90) / 66) ** 2 + ((yy - 132) / 118) ** 2) <= 1
        clean.putalpha(Image.fromarray((silhouette * 255).astype(np.uint8), "L"))
        damaged = clean.copy()
        alpha = np.asarray(damaged.getchannel("A")).copy()
        for center_x, center_y in ((62, 58), (91, 72), (119, 87), (71, 112), (105, 128), (132, 145)):
            hole = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 11 ** 2
            alpha[hole] = 0
        damaged.putalpha(Image.fromarray(alpha, "L"))
        self.assertGreater(app_module.cutout_quality(clean), 0.25)
        self.assertLess(app_module.cutout_quality(damaged), 0.20)


if __name__ == "__main__":
    unittest.main()
