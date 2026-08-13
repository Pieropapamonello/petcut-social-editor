import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from layered_effects import (
    BACKGROUND_NAMES,
    HEIGHT,
    SOURCE_PROXY_MAX_SIDE,
    WIDTH,
    prepare_layered_assets,
    render_layered_board,
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(root: Path) -> tuple[Path, Path]:
    source = Image.new("RGB", (360, 540), (22, 66, 126))
    draw = ImageDraw.Draw(source)
    for y in range(0, 540, 36):
        draw.rectangle((0, y, 359, y + 17), fill=(30 + y // 8, 118, 74 + y // 10))
    draw.ellipse((48, 80, 315, 350), fill=(234, 164, 52))
    source_path = root / "source.png"
    source.save(source_path)

    subject = Image.new("RGBA", (230, 350), (0, 0, 0, 0))
    draw = ImageDraw.Draw(subject)
    draw.ellipse((34, 38, 196, 218), fill=(224, 38, 55, 255))
    draw.polygon(((60, 76), (30, 4), (103, 52)), fill=(224, 38, 55, 255))
    draw.polygon(((130, 52), (201, 3), (178, 86)), fill=(224, 38, 55, 255))
    draw.rounded_rectangle((56, 170, 178, 320), radius=48, fill=(205, 27, 45, 255))
    draw.ellipse((75, 305, 117, 345), fill=(205, 27, 45, 255))
    draw.ellipse((135, 305, 177, 345), fill=(205, 27, 45, 255))
    subject_path = root / "subject.png"
    subject.save(subject_path)
    return source_path, subject_path


class LayeredEffectsTests(unittest.TestCase):
    def test_assets_are_sized_cached_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            first = prepare_layered_assets(source, subject, root / "cache", seed=31)
            digests = {name: file_digest(path) for name, path in first.backgrounds.items()}
            second = prepare_layered_assets(source, subject, root / "cache", seed=31)

            self.assertEqual(first.root, second.root)
            self.assertEqual(set(first.backgrounds), set(BACKGROUND_NAMES))
            self.assertEqual(digests, {name: file_digest(path) for name, path in second.backgrounds.items()})
            for path in first.backgrounds.values():
                with Image.open(path) as plate:
                    self.assertEqual(plate.size, (WIDTH, HEIGHT))
            with Image.open(first.subject_layer) as layer, Image.open(first.contact_shadow) as shadow:
                self.assertEqual(layer.size, (WIDTH, HEIGHT))
                self.assertEqual(shadow.size, (WIDTH, HEIGHT))

    def test_background_plates_are_visibly_different(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            assets = prepare_layered_assets(source, subject, root / "cache")
            arrays = {
                name: np.asarray(Image.open(path).convert("RGB").resize((72, 128)), dtype=np.int16)
                for name, path in assets.backgrounds.items()
            }
            baseline = arrays["blurred_zoom"]
            differences = [np.mean(np.abs(baseline - arrays[name])) for name in BACKGROUND_NAMES[1:]]
            self.assertTrue(all(value > 8.0 for value in differences), differences)

    def test_large_source_is_bounded_before_plate_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, subject = fixtures(root)
            large = Image.new("RGB", (3200, 1800), (38, 94, 146))
            ImageDraw.Draw(large).rectangle((300, 250, 2900, 1550), fill=(181, 92, 47))
            source = root / "large-source.jpg"
            large.save(source, quality=88)
            del large

            assets = prepare_layered_assets(source, subject, root / "cache")
            manifest = json.loads(assets.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_size_original"], [3200, 1800])
            self.assertLessEqual(max(manifest["source_size_proxy"]), SOURCE_PROXY_MAX_SIDE)
            self.assertEqual(manifest["source_proxy_max_side"], SOURCE_PROXY_MAX_SIDE)
            for path in assets.backgrounds.values():
                with Image.open(path) as plate:
                    self.assertEqual(plate.size, (WIDTH, HEIGHT))

    def test_master_preserves_real_alpha_and_discards_specks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            # A disconnected alpha speck represents leftover background.
            with Image.open(subject).convert("RGBA") as image:
                image.putpixel((225, 345), (255, 255, 255, 255))
                image.save(subject)
            assets = prepare_layered_assets(source, subject, root / "cache")
            with Image.open(assets.subject_sprite).convert("RGBA") as master:
                alpha = np.asarray(master.getchannel("A"))
                self.assertGreater(np.count_nonzero(alpha == 0), 0)
                self.assertGreater(np.count_nonzero(alpha == 255), 1000)
                self.assertLess(np.count_nonzero(alpha > 0), alpha.size * 0.82)

    def test_frames_change_while_foreground_stays_central(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            assets = prepare_layered_assets(source, subject, root / "cache")
            paths = [
                render_layered_board(
                    assets,
                    shot_index=index,
                    progress=0.18 + index * 0.31,
                    accent=1.0 if index == 1 else 0.2,
                )
                for index in range(3)
            ]
            frames = [np.asarray(Image.open(path).convert("RGBA")) for path in paths]
            self.assertTrue(all(frame.shape == (HEIGHT, WIDTH, 4) for frame in frames))
            self.assertGreater(np.mean(np.abs(frames[0][:, :, :3].astype(int) - frames[1][:, :, :3].astype(int))), 12)

            for frame in frames:
                # The synthetic master is the only strongly red layer; RGB
                # ghosts are ignored by requiring red to dominate both other
                # channels.  Its centroid must stay in the central safe area.
                red = (
                    (frame[:, :, 0] > 150)
                    & (frame[:, :, 0] > frame[:, :, 1] * 1.65)
                    & (frame[:, :, 0] > frame[:, :, 2] * 1.45)
                )
                yy, xx = np.nonzero(red)
                self.assertGreater(len(xx), 4000)
                self.assertLess(abs(float(xx.mean()) - WIDTH / 2), WIDTH * 0.12)
                self.assertLess(abs(float(yy.mean()) - HEIGHT * 0.53), HEIGHT * 0.17)

    def test_same_request_reuses_identical_cached_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            assets = prepare_layered_assets(source, subject, root / "cache", seed=5)
            first = render_layered_board(assets, shot_index=7, progress=0.375, accent=0.8)
            digest = file_digest(first)
            second = render_layered_board(assets, shot_index=7, progress=0.375, accent=0.8)
            self.assertEqual(first, second)
            self.assertEqual(digest, file_digest(second))

    def test_full_frame_subject_alpha_is_inpainted_from_every_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (360, 540), (18, 132, 48))
            draw = ImageDraw.Draw(source)
            # The red shape is the original subject burned into the source.
            draw.rounded_rectangle((100, 105, 260, 425), radius=55, fill=(244, 18, 31))
            source_path = root / "source-with-subject.png"
            source.save(source_path)

            full_frame_subject = Image.new("RGBA", source.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(full_frame_subject)
            draw.rounded_rectangle((100, 105, 260, 425), radius=55, fill=(244, 18, 31, 255))
            subject_path = root / "registered-subject.png"
            full_frame_subject.save(subject_path)

            assets = prepare_layered_assets(source_path, subject_path, root / "cache")
            manifest = json.loads(assets.manifest.read_text(encoding="utf-8"))
            self.assertTrue(manifest["subject_removed"])
            self.assertEqual(manifest["matte_origin"], "subject_alpha")

            for name, path in assets.backgrounds.items():
                plate = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
                # Plate transforms can introduce purple accents, but the
                # large, strongly red original silhouette must be gone.
                original_red = (
                    (plate[:, :, 0] > 175)
                    & (plate[:, :, 0] > plate[:, :, 1] * 1.8)
                    & (plate[:, :, 0] > plate[:, :, 2] * 1.5)
                )
                self.assertLess(float(original_red.mean()), 0.003, name)

    def test_explicit_matte_aliases_and_mismatched_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            with Image.open(source) as opened:
                source_size = opened.size
            mask = Image.new("L", source_size, 0)
            ImageDraw.Draw(mask).ellipse((48, 80, 315, 350), fill=255)

            via_alpha = prepare_layered_assets(
                source,
                subject,
                root / "cache",
                source_alpha=mask,
            )
            via_matte = prepare_layered_assets(
                source,
                subject,
                root / "cache",
                full_frame_matte=np.asarray(mask),
            )
            self.assertEqual(via_alpha.cache_key, via_matte.cache_key)

            baseline = prepare_layered_assets(source, subject, root / "cache", seed=9)
            mismatch = prepare_layered_assets(
                source,
                subject,
                root / "cache",
                seed=9,
                full_frame_matte=Image.new("L", (10, 10), 255),
            )
            self.assertEqual(baseline.cache_key, mismatch.cache_key)
            self.assertEqual(
                {name: file_digest(path) for name, path in baseline.backgrounds.items()},
                {name: file_digest(path) for name, path in mismatch.backgrounds.items()},
            )

    def test_unknown_background_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, subject = fixtures(root)
            assets = prepare_layered_assets(source, subject, root / "cache")
            with self.assertRaises(ValueError):
                render_layered_board(assets, background="not-a-plate")


if __name__ == "__main__":
    unittest.main()
