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
from subject_pipeline import matte_diagnostics


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
        with app_module.AUDIO_ANALYSIS_LOCK:
            app_module.AUDIO_ANALYSIS_CACHE.clear()
        self.client = app.test_client()

    def test_identical_audio_preview_and_render_reuse_analysis(self):
        rhythm = {"bpm": 120.0, "edit_bpm": 120.0, "duration": 6.0}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "preview.wav"
            second = Path(directory) / "render.wav"
            first.write_bytes(b"same-audio-content")
            second.write_bytes(b"same-audio-content")
            with patch.object(
                app_module, "analyze_audio_structure", return_value=rhythm
            ) as analyze:
                preview = app_module.analyze_audio_cached(first)
                preview["bpm"] = 1
                final = app_module.analyze_audio_cached(second)
            analyze.assert_called_once_with(first, 0.0)
            self.assertEqual(final["bpm"], 120.0)

    def test_home_and_health_expose_single_reference_mode(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Il montaggio", page.data)
        self.assertIn(b"Soggetto principale", page.data)
        self.assertIn(b'name="primary"', page.data)
        self.assertNotIn(b'name="style"', page.data)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        body = health.get_json()
        self.assertEqual(body["profile"], "subject-sync-3d-v1")
        self.assertEqual(body["output"], "576x1024")
        self.assertEqual(body["mode"], "reference_edit")
        self.assertNotIn("presets", body)

    def test_audio_analysis_contract(self):
        response = self.client.post(
            "/api/analyze-audio",
            data={
                "audio": (click_track(duration=12.0), "song.wav"),
                "duration": "6",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreater(body["bpm"], 0)
        self.assertGreater(body["edit_bpm"], 0)
        self.assertGreater(body["duration"], 4.0)
        self.assertLessEqual(body["duration"], 6.0)
        self.assertAlmostEqual(body["audio_duration"], 12.0, delta=0.1)
        self.assertGreaterEqual(body["audio_start"], 0.0)
        self.assertAlmostEqual(
            body["audio_end"] - body["audio_start"], body["duration"], delta=0.05
        )
        self.assertGreaterEqual(body["drop_time"], 0.0)
        self.assertLessEqual(body["drop_time"], body["duration"])
        self.assertGreater(body["visual_cuts"], 5)
        self.assertGreaterEqual(body["recommended_content"], 1)
        self.assertTrue(body["sections"])
        self.assertIn("Estratto scelto", body["message"])

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

    def test_dark_subject_detail_is_lifted_inside_matte_only(self):
        rgb = np.full((90, 80, 3), (205, 180, 145), np.uint8)
        alpha = np.zeros((90, 80), np.uint8)
        cv2 = app_module.cv2
        cv2.ellipse(alpha, (40, 46), (22, 34), 0, 0, 360, 255, -1)
        rgb[alpha > 0] = (9, 12, 15)
        lifted = app_module._lift_subject_detail(rgb, alpha)
        before = float(np.mean(app_module.cv2.cvtColor(rgb, app_module.cv2.COLOR_RGB2GRAY)[alpha > 200]))
        after = float(np.mean(app_module.cv2.cvtColor(lifted, app_module.cv2.COLOR_RGB2GRAY)[alpha > 200]))
        self.assertGreater(after, before + 35)
        self.assertTrue(np.array_equal(lifted[alpha == 0], rgb[alpha == 0]))

    def test_cutouts_are_sampled_only_from_declared_protagonist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary.jpg"
            extra = root / "background.jpg"
            primary.write_bytes(b"primary")
            extra.write_bytes(b"extra")
            sampled_paths = []

            def fake_frame(path, index, count):
                sampled_paths.append(Path(path))
                return Image.new("RGB", (120, 160), (95 + index, 72, 55))

            yy, xx = np.mgrid[:160, :120]
            alpha = (
                (((xx - 60) / 42) ** 2 + ((yy - 78) / 66) ** 2 <= 1) * 255
            ).astype(np.uint8)

            def fake_isolate(frame):
                result = frame.convert("RGBA")
                result.putalpha(Image.fromarray(alpha, "L"))
                return result

            def fake_refine(rgb, coarse_alpha, _bbox):
                return rgb, coarse_alpha, matte_diagnostics(coarse_alpha)

            with (
                patch.object(app_module, "source_frame", side_effect=fake_frame),
                patch.object(app_module, "isolate_subject", side_effect=fake_isolate),
                patch.object(app_module, "refine_cutout", side_effect=fake_refine),
            ):
                cutouts, samples = app_module.create_cutouts(
                    [primary, extra], root, 3
                )

            self.assertEqual(sampled_paths, [primary, primary, primary])
            self.assertTrue(cutouts)
            self.assertTrue(samples)
            self.assertEqual(len(list(root.glob("sample-*-alpha.png"))), 3)
            for cutout in cutouts:
                with Image.open(cutout) as image:
                    self.assertEqual(image.mode, "RGBA")
                    diagnostics = matte_diagnostics(np.asarray(image.getchannel("A")))
                    self.assertNotIn("background-rectangle", diagnostics.reasons)

    def test_instance_miss_uses_saliency_before_grabcut(self):
        frame = Image.new("RGB", (80, 100), (65, 80, 105))
        yy, xx = np.mgrid[:100, :80]
        saliency = np.exp(-(((xx - 40) / 24) ** 2 + ((yy - 48) / 38) ** 2)).astype(
            np.float32
        )
        with (
            patch.object(app_module, "instance_subject_mask", return_value=(None, None)),
            patch.object(app_module, "release_instance_after_miss") as release,
            patch.object(app_module, "neural_saliency", return_value=saliency) as neural,
        ):
            result = app_module.isolate_subject(frame)
        release.assert_called_once_with()
        neural.assert_called_once()
        self.assertEqual(result.mode, "RGBA")
        self.assertIsNotNone(result.getchannel("A").getbbox())

    def test_render_job_analyses_whole_song_and_persists_excerpt_metadata(self):
        rhythm = {
            "bpm": 90.0,
            "edit_bpm": 180.0,
            "duration": 30.0,
            "drop_time": 20.5,
            "onsets": [],
            "onset_strengths": [],
            "beat_times": [],
            "sections": [],
            "energy_curve": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary.jpg"
            extra = root / "background.jpg"
            audio = root / "song.wav"
            primary.write_bytes(b"image")
            extra.write_bytes(b"image")
            audio.write_bytes(b"audio")
            job_id = "render-unit"
            app_module.JOBS[job_id] = {"status": "processing", "progress": 1}
            metadata = {
                "duration": 18.1,
                "audio_start": 11.9,
                "audio_end": 30.0,
                "source_drop": 20.5,
                "drop_time": 8.6,
                "scenes": 42,
            }
            with (
                patch.object(app_module, "analyze_audio_structure", return_value=rhythm) as analyse,
                patch.object(app_module, "create_cutouts", return_value=([primary], [primary])) as cutouts,
                patch.object(app_module, "release_subject_detector"),
                patch.object(app_module, "render_preset", return_value=metadata) as render,
                patch.object(app_module, "keep_only_output"),
            ):
                app_module.render_job(
                    job_id, [primary, extra], audio, "LUNA", 24, root
                )

            analyse.assert_called_once_with(audio, 0.0)
            cutouts.assert_called_once_with([primary, extra], root, 1)
            self.assertEqual(render.call_args.args[5], 24.0)
            job = app_module.JOBS.pop(job_id)
            self.assertEqual(job["status"], "complete")
            self.assertEqual(job["duration"], 18.1)
            self.assertEqual(job["audio_start"], 11.9)
            self.assertEqual(job["audio_end"], 30.0)
            self.assertEqual(job["source_drop"], 20.5)


if __name__ == "__main__":
    unittest.main()
