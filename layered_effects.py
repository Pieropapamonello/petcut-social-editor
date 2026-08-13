"""Deterministic, lightweight 2.5D compositing for PetCut.

The module deliberately has no dependency on the web application or the
timeline renderer.  It prepares reusable background plates and a canonical
subject sprite once, then renders inexpensive 576x1024 boards from those
cached assets.  Pillow performs most operations; OpenCV is used only for
connected-component cleanup and the small perspective warp.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


WIDTH = 576
HEIGHT = 1024
SOURCE_PROXY_MAX_SIDE = 1280
ASSET_VERSION = "layered-v3"
BACKGROUND_NAMES = (
    "blurred_zoom",
    "duotone",
    "radial_rays",
    "split_panels",
    "mirror_tunnel",
    "gradient",
)

_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class LayeredAssets:
    """Paths to the reusable layers for one source/subject pair."""

    root: Path
    cache_key: str
    backgrounds: Mapping[str, Path]
    subject_sprite: Path
    subject_layer: Path
    contact_shadow: Path
    manifest: Path
    width: int = WIDTH
    height: int = HEIGHT


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _hash_file(path: Path, digest: "hashlib._Hash") -> None:
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)


def _asset_key(source: Path, subject: Path, seed: int, full_frame_matte: Image.Image | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(f"{ASSET_VERSION}:{WIDTH}x{HEIGHT}:{int(seed)}".encode("ascii"))
    _hash_file(source, digest)
    _hash_file(subject, digest)
    if full_frame_matte is None:
        digest.update(b":no-full-frame-matte")
    else:
        matte = full_frame_matte.convert("L")
        digest.update(f":matte:{matte.width}x{matte.height}:".encode("ascii"))
        digest.update(np.asarray(matte, dtype=np.uint8).tobytes())
    return digest.hexdigest()[:20]


def _atomic_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _open_oriented(path: Path, mode: str) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert(mode)


def _open_source_proxy(
    path: Path, max_side: int = SOURCE_PROXY_MAX_SIDE
) -> tuple[Image.Image, tuple[int, int]]:
    """Decode an oriented scene into a bounded plate-generation proxy.

    JPEG ``draft`` asks the decoder for a reduced native resolution before a
    pixel buffer is materialised.  The final thumbnail is an unconditional
    guard for PNG/WebP and for JPEG dimensions between draft levels.
    """

    with Image.open(path) as opened:
        raw_width, raw_height = opened.size
        try:
            orientation = int(opened.getexif().get(274, 1))
        except (TypeError, ValueError):
            orientation = 1
        original_size = (
            (raw_height, raw_width)
            if orientation in {5, 6, 7, 8}
            else (raw_width, raw_height)
        )
        try:
            opened.draft("RGB", (max_side, max_side))
        except (AttributeError, OSError):
            pass
        proxy = ImageOps.exif_transpose(opened)
        proxy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return proxy.convert("RGB"), original_size


def _matte_from_value(value: str | Path | Image.Image | np.ndarray) -> Image.Image | None:
    """Read a caller-supplied mask without guessing its placement."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return None
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            if "A" in image.getbands():
                return image.getchannel("A").copy()
            return image.convert("L")
    if isinstance(value, Image.Image):
        image = ImageOps.exif_transpose(value)
        if "A" in image.getbands():
            return image.getchannel("A").copy()
        return image.convert("L")
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, 3]
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            return None
    if array.ndim != 2 or not array.size:
        return None
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return Image.fromarray(np.nan_to_num(array, nan=0.0).clip(0, 255).astype(np.uint8), "L")


def _usable_full_frame_matte(matte: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    """Accept only a meaningful mask already registered to the source."""

    if matte is None or matte.size != size:
        return None
    result = matte.convert("L")
    alpha = np.asarray(result, dtype=np.uint8)
    coverage = float(np.count_nonzero(alpha > 8)) / max(1, alpha.size)
    # A nearly empty matte cannot remove a useful subject; a nearly opaque
    # matte is normally an ordinary RGB image converted to alpha by mistake.
    if coverage < 0.0005 or coverage > 0.88 or int(alpha.max()) <= 8:
        return None
    return result


def _subject_full_frame_alpha(subject_path: Path, size: tuple[int, int]) -> Image.Image | None:
    """Recover the registered alpha when the subject is a full-frame PNG."""

    try:
        with Image.open(subject_path) as opened:
            subject = ImageOps.exif_transpose(opened)
            if "A" not in subject.getbands() or subject.size != size:
                return None
            return _usable_full_frame_matte(subject.getchannel("A").copy(), size)
    except (OSError, ValueError):
        return None


def _resolve_full_frame_matte(
    source_size: tuple[int, int],
    subject_path: Path,
    source_alpha: str | Path | Image.Image | np.ndarray | None,
    full_frame_matte: str | Path | Image.Image | np.ndarray | None,
) -> tuple[Image.Image | None, str | None]:
    # ``full_frame_matte`` is the descriptive spelling; ``source_alpha`` is
    # retained as a convenient alias for callers that already expose alpha.
    explicit = full_frame_matte if full_frame_matte is not None else source_alpha
    if explicit is not None:
        resolved = _usable_full_frame_matte(_matte_from_value(explicit), source_size)
        return resolved, "explicit" if resolved is not None else None
    automatic = _subject_full_frame_alpha(subject_path, source_size)
    return automatic, "subject_alpha" if automatic is not None else None


def _inpaint_subject(source: Image.Image, matte: Image.Image) -> Image.Image:
    """Remove the registered foreground before any stylised plate is made."""

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    alpha = np.asarray(matte.convert("L"), dtype=np.uint8)
    longest = max(source.size)
    # TELEA temporarily allocates several image-sized buffers.  The generated
    # plate is only 576x1024, so cap this optional path for small Render plans.
    if longest > 1600:
        ratio = 1600.0 / longest
        size = (max(2, round(source.width * ratio)), max(2, round(source.height * ratio)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    height, width = alpha.shape
    radius = max(2, min(10, round(min(width, height) * 0.006)))
    kernel_size = radius * 2 + 1
    mask = np.where(alpha > 5, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    clean = cv2.inpaint(bgr, mask, float(max(3, radius)), cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB), "RGB")


def _cover(image: Image.Image, width: int = WIDTH, height: int = HEIGHT, scale: float = 1.0) -> Image.Image:
    target = (max(width, round(width * scale)), max(height, round(height * scale)))
    fitted = ImageOps.fit(image.convert("RGB"), target, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if target == (width, height):
        return fitted
    left = (fitted.width - width) // 2
    top = (fitted.height - height) // 2
    return fitted.crop((left, top, left + width, top + height))


def _palette(image: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    pixels = np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1, 3)
    luminance = pixels @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    low = pixels[np.argsort(luminance)[len(pixels) // 5]]
    high = pixels[np.argsort(luminance)[len(pixels) * 4 // 5]]
    mean = pixels.mean(axis=0)
    dark = np.clip(low * 0.22 + np.asarray((3, 5, 12)), 0, 255)
    light = np.clip(high * 1.08 + 18, 0, 255)
    accent = np.clip(np.roll(mean, 1) * 1.15 + np.asarray((16, 5, 24)), 0, 255)
    # Convert NumPy scalars to Python ints.  Pillow's colorize arithmetic can
    # otherwise overflow uint8 values while constructing its lookup table.
    return (
        tuple(int(value) for value in dark),
        tuple(int(value) for value in light),
        tuple(int(value) for value in accent),
    )


def _vignette(image: Image.Image, strength: float = 0.36) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    yy, xx = np.ogrid[-1.0:1.0:complex(0, image.height), -1.0:1.0:complex(0, image.width)]
    radius = np.sqrt(xx * xx + yy * yy)
    multiplier = np.clip(1.0 - strength * np.power(radius, 1.45), 0.48, 1.0)
    rgb *= multiplier[:, :, None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _blurred_zoom(base: Image.Image) -> Image.Image:
    result = _cover(base, scale=1.14).filter(ImageFilter.GaussianBlur(17))
    result = ImageEnhance.Color(result).enhance(0.78)
    result = ImageEnhance.Contrast(result).enhance(1.12)
    result = ImageEnhance.Brightness(result).enhance(0.62)
    return _vignette(result, 0.30)


def _duotone(base: Image.Image, palette: tuple[tuple[int, int, int], ...]) -> Image.Image:
    gray = ImageOps.grayscale(_cover(base).filter(ImageFilter.GaussianBlur(3)))
    result = ImageOps.colorize(gray, black=palette[0], white=palette[2])
    return _vignette(ImageEnhance.Contrast(result).enhance(1.22), 0.26)


def _radial_rays(base: Image.Image, palette: tuple[tuple[int, int, int], ...], seed: int) -> Image.Image:
    result = _duotone(base, palette).convert("RGBA")
    result = Image.blend(
        result,
        Image.new("RGBA", result.size, (*palette[0], 255)),
        0.20,
    )
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    centre = (WIDTH // 2, round(HEIGHT * 0.51))
    radius = math.hypot(WIDTH, HEIGHT) * 1.15
    offset = (int(seed) % 23) * math.pi / 180
    ray_count = 24
    for index in range(0, ray_count, 2):
        first = offset + index * 2 * math.pi / ray_count
        second = offset + (index + 1) * 2 * math.pi / ray_count
        points = [
            centre,
            (centre[0] + radius * math.cos(first), centre[1] + radius * math.sin(first)),
            (centre[0] + radius * math.cos(second), centre[1] + radius * math.sin(second)),
        ]
        color = palette[2] if index % 4 == 0 else palette[1]
        draw.polygon(points, fill=(*color, 94 if index % 4 == 0 else 64))
    overlay = overlay.filter(ImageFilter.GaussianBlur(1.2))
    return Image.alpha_composite(result, overlay).convert("RGB")


def _split_panels(base: Image.Image) -> Image.Image:
    source = _cover(base, scale=1.16)
    result = Image.new("RGB", (WIDTH, HEIGHT), (7, 8, 13))
    gap = 6
    panel_width = (WIDTH - gap * 3) // 4
    for index in range(4):
        left = round(index * source.width / 8)
        panel = source.crop((left, 0, min(source.width, left + panel_width + 70), HEIGHT))
        panel = ImageOps.fit(panel, (panel_width, HEIGHT), Image.Resampling.LANCZOS)
        if index % 2:
            panel = ImageOps.mirror(panel)
        panel = ImageEnhance.Brightness(panel).enhance(0.66 + index * 0.045)
        result.paste(panel, (index * (panel_width + gap), 0))
    return _vignette(result, 0.23)


def _mirror_tunnel(base: Image.Image) -> Image.Image:
    source = _cover(base)
    result = source.filter(ImageFilter.GaussianBlur(8))
    result = ImageEnhance.Brightness(result).enhance(0.45)
    draw = ImageDraw.Draw(result)
    for index, scale in enumerate((0.88, 0.68, 0.50, 0.34)):
        size = (round(WIDTH * scale), round(HEIGHT * scale))
        panel = ImageOps.fit(source, size, Image.Resampling.LANCZOS)
        if index % 2:
            panel = ImageOps.mirror(panel)
        panel = ImageEnhance.Brightness(panel).enhance(0.82 - index * 0.08)
        position = ((WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2)
        draw.rounded_rectangle(
            (position[0] - 4, position[1] - 4, position[0] + size[0] + 3, position[1] + size[1] + 3),
            radius=max(8, 28 - index * 5),
            fill=(5, 6, 10),
        )
        result.paste(panel, position)
    return _vignette(result, 0.31)


def _gradient(base: Image.Image, palette: tuple[tuple[int, int, int], ...]) -> Image.Image:
    top = np.asarray(palette[0], dtype=np.float32)
    middle = np.asarray(palette[2], dtype=np.float32)
    bottom = np.asarray(palette[1], dtype=np.float32) * 0.48
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None, None]
    first = top[None, None, :] * (1 - y * 2) + middle[None, None, :] * (y * 2)
    second = middle[None, None, :] * (2 - y * 2) + bottom[None, None, :] * (y * 2 - 1)
    rgb = np.where((y <= 0.5), first, second)
    rgb = np.repeat(rgb, WIDTH, axis=1)
    result = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    texture = _cover(base).filter(ImageFilter.GaussianBlur(22))
    texture = ImageEnhance.Contrast(ImageOps.grayscale(texture)).enhance(1.4).convert("RGB")
    return _vignette(Image.blend(result, texture, 0.14), 0.25)


def _clean_subject(subject: Image.Image) -> Image.Image:
    rgba = np.asarray(subject.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3]
    binary = (alpha > 8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("Il soggetto non contiene un canale alpha visibile")
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    largest_area = int(stats[largest, cv2.CC_STAT_AREA])
    keep = labels == largest
    # Keep meaningful detached details (a paw separated by a weak alpha edge),
    # but discard specks and disconnected pieces of the old background.
    for component in range(1, count):
        if component != largest and stats[component, cv2.CC_STAT_AREA] >= max(24, largest_area * 0.025):
            keep |= labels == component
    alpha[~keep] = 0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # Pull edge colour from confident interior pixels.  This reduces white or
    # cyan spill without expanding the matte into its former background.
    interior = (alpha >= 220).astype(np.float32)
    weight = cv2.GaussianBlur(interior, (0, 0), 3.0)
    for channel in range(3):
        numerator = cv2.GaussianBlur(rgba[:, :, channel].astype(np.float32) * interior, (0, 0), 3.0)
        replacement = numerator / np.maximum(weight, 1e-4)
        edge = (alpha > 0) & (alpha < 245) & (weight > 0.025)
        rgba[:, :, channel][edge] = np.clip(
            rgba[:, :, channel][edge] * 0.35 + replacement[edge] * 0.65,
            0,
            255,
        ).astype(np.uint8)
        rgba[:, :, channel][alpha == 0] = 0
    rgba[:, :, 3] = alpha
    cleaned = Image.fromarray(rgba, "RGBA")
    bounds = cleaned.getchannel("A").getbbox()
    if not bounds:
        raise ValueError("Il soggetto non contiene pixel visibili")
    cleaned = cleaned.crop(bounds)
    cleaned.thumbnail((520, 820), Image.Resampling.LANCZOS)
    # Recover texture softened by detector upsampling and alpha refinement.
    # Sharpening the canonical sprite once is cleaner than sharpening every
    # background and makes dark fur readable without thickening the matte.
    cleaned = ImageEnhance.Sharpness(cleaned).enhance(1.55)
    return cleaned


def _fit_sprite(sprite: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / max(1, sprite.width), max_height / max(1, sprite.height))
    size = (max(1, round(sprite.width * ratio)), max(1, round(sprite.height * ratio)))
    return sprite.resize(size, Image.Resampling.LANCZOS)


def _place_layer(sprite: Image.Image, centre_x: float, bottom: float) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    x = round(centre_x - sprite.width / 2)
    y = round(bottom - sprite.height)
    layer.alpha_composite(sprite, dest=(x, y))
    return layer


def _shadow_layer(subject_width: int, subject_height: int, centre_x: float, bottom: float, accent: float = 0.0) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    shadow_width = max(52, round(subject_width * (0.64 + 0.08 * accent)))
    shadow_height = max(18, round(subject_height * 0.055))
    x = round(centre_x - shadow_width / 2)
    y = round(bottom - shadow_height * 0.48)
    draw.ellipse((x, y, x + shadow_width, y + shadow_height), fill=(0, 0, 0, round(92 + 38 * accent)))
    return layer.filter(ImageFilter.GaussianBlur(max(7, round(shadow_height * 0.55))))


def _canonical_layers(sprite: Image.Image) -> tuple[Image.Image, Image.Image]:
    fitted = _fit_sprite(sprite, round(WIDTH * 0.76), round(HEIGHT * 0.59))
    bottom = HEIGHT * 0.82
    return (
        _place_layer(fitted, WIDTH / 2, bottom),
        _shadow_layer(fitted.width, fitted.height, WIDTH / 2, bottom),
    )


def _paths_for(root: Path) -> tuple[dict[str, Path], Path, Path, Path, Path]:
    backgrounds = {name: root / f"background-{name}.png" for name in BACKGROUND_NAMES}
    return (
        backgrounds,
        root / "subject-master.png",
        root / "subject-layer.png",
        root / "contact-shadow.png",
        root / "manifest.json",
    )


def prepare_layered_assets(
    source_image: str | Path,
    subject_sprite: str | Path,
    cache_dir: str | Path,
    *,
    seed: int = 0,
    source_alpha: str | Path | Image.Image | np.ndarray | None = None,
    full_frame_matte: str | Path | Image.Image | np.ndarray | None = None,
) -> LayeredAssets:
    """Create or reuse all plates required by :func:`render_layered_board`.

    ``source_image`` is the full scene; ``subject_sprite`` must be a PNG (or
    another format) with a useful alpha channel.  ``source_alpha`` and
    ``full_frame_matte`` are aliases for an optional mask registered to the
    full source. If neither is supplied, a full-frame alpha in
    ``subject_sprite`` is discovered automatically. A mismatched/cropped mask
    is ignored, preserving the original background behaviour. The returned
    cache folder is content-addressed, so repeated jobs do not regenerate the
    same assets.
    """

    source_path, subject_path = Path(source_image), Path(subject_sprite)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not subject_path.is_file():
        raise FileNotFoundError(subject_path)
    source, original_source_size = _open_source_proxy(source_path)
    matte, matte_origin = _resolve_full_frame_matte(
        original_source_size,
        subject_path,
        source_alpha,
        full_frame_matte,
    )
    if matte is not None and matte.size != source.size:
        matte = matte.resize(source.size, Image.Resampling.BILINEAR)
    key = _asset_key(source_path, subject_path, seed, matte)
    root = Path(cache_dir) / f"layers-{key}"
    backgrounds, master_path, layer_path, shadow_path, manifest_path = _paths_for(root)
    expected = [*backgrounds.values(), master_path, layer_path, shadow_path, manifest_path]

    with _CACHE_LOCK:
        if not all(path.is_file() for path in expected):
            root.mkdir(parents=True, exist_ok=True)
            base = _inpaint_subject(source, matte) if matte is not None else source
            palette = _palette(base)
            plates = {
                "blurred_zoom": _blurred_zoom(base),
                "duotone": _duotone(base, palette),
                "radial_rays": _radial_rays(base, palette, seed),
                "split_panels": _split_panels(base),
                "mirror_tunnel": _mirror_tunnel(base),
                "gradient": _gradient(base, palette),
            }
            for name, plate in plates.items():
                _atomic_png(plate.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGB"), backgrounds[name])

            master = _clean_subject(_open_oriented(subject_path, "RGBA"))
            subject_layer, shadow = _canonical_layers(master)
            _atomic_png(master, master_path)
            _atomic_png(subject_layer, layer_path)
            _atomic_png(shadow, shadow_path)
            manifest = {
                "version": ASSET_VERSION,
                "cache_key": key,
                "size": [WIDTH, HEIGHT],
                "source_size_original": list(original_source_size),
                "source_size_proxy": list(source.size),
                "source_proxy_max_side": SOURCE_PROXY_MAX_SIDE,
                "seed": int(seed),
                "backgrounds": list(BACKGROUND_NAMES),
                "subject_removed": matte is not None,
                "matte_origin": matte_origin,
            }
            temporary = manifest_path.with_name(
                f".{manifest_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temporary.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, manifest_path)

    return LayeredAssets(
        root=root,
        cache_key=key,
        backgrounds=backgrounds,
        subject_sprite=master_path,
        subject_layer=layer_path,
        contact_shadow=shadow_path,
        manifest=manifest_path,
    )


def _moving_background(plate: Image.Image, progress: float, shot_index: int, accent: float) -> Image.Image:
    phase = math.radians((shot_index * 137 + 29) % 360)
    eased = progress * progress * (3.0 - 2.0 * progress)
    travel = eased * 2.0 - 1.0
    zoom = 1.075 + 0.027 * math.sin(progress * math.pi) + accent * 0.018 * (1.0 - progress)
    width, height = round(WIDTH * zoom), round(HEIGHT * zoom)
    enlarged = plate.resize((width, height), Image.Resampling.BICUBIC)
    available_x, available_y = width - WIDTH, height - HEIGHT
    shift_x = math.cos(phase) * available_x * 0.43 * travel
    shift_y = math.sin(phase) * available_y * 0.34 * travel
    left = round(available_x / 2 + shift_x)
    top = round(available_y / 2 + shift_y)
    left = max(0, min(available_x, left))
    top = max(0, min(available_y, top))
    return enlarged.crop((left, top, left + WIDTH, top + HEIGHT)).convert("RGBA")


def _perspective_sprite(sprite: Image.Image, tilt: float) -> Image.Image:
    tilt = _clamp(tilt, -1.0, 1.0)
    rgba = np.asarray(sprite.convert("RGBA"), dtype=np.uint8)
    height, width = rgba.shape[:2]
    padding = max(8, round(max(width, height) * 0.11))
    padded = cv2.copyMakeBorder(rgba, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    canvas_height, canvas_width = padded.shape[:2]
    source = np.float32(
        [[padding, padding], [padding + width, padding], [padding + width, padding + height], [padding, padding + height]]
    )
    top_shift = tilt * width * 0.065
    bottom_shift = -tilt * width * 0.025
    pinch = abs(tilt) * width * 0.035
    destination = np.float32(
        [
            [padding + pinch + top_shift, padding],
            [padding + width - pinch + top_shift, padding],
            [padding + width + bottom_shift, padding + height],
            [padding + bottom_shift, padding + height],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        padded,
        matrix,
        (canvas_width, canvas_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    result = Image.fromarray(warped, "RGBA")
    bounds = result.getchannel("A").getbbox()
    return result.crop(bounds) if bounds else sprite.copy()


def _tinted(sprite: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    rgba = np.asarray(sprite.convert("RGBA"), dtype=np.uint8).copy()
    visible = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)[None, None, :]
    rgba[:, :, :3] = np.clip(rgb * 0.30 + tint * 0.70, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(visible * 255 * opacity, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _rim(sprite: Image.Image, opacity: float) -> Image.Image:
    alpha = sprite.getchannel("A")
    expanded = alpha.filter(ImageFilter.MaxFilter(7))
    border = ImageChops.subtract(expanded, alpha)
    border = border.point(lambda value: round(value * opacity))
    layer = Image.new("RGBA", sprite.size, (105, 215, 255, 0))
    layer.putalpha(border)
    return layer


def render_layered_board(
    assets: LayeredAssets,
    destination: str | Path | None = None,
    *,
    shot_index: int = 0,
    progress: float = 0.0,
    accent: float | bool = 0.0,
    background: str | None = None,
) -> Path:
    """Render one deterministic 2.5D board with a persistent central subject.

    ``progress`` is local to the shot (0..1), while ``accent`` controls scale
    overshoot, trails and RGB separation.  If no destination is supplied the
    composed frame itself is cached beside the prepared layers.
    """

    progress = _clamp(progress)
    accent_value = _clamp(float(accent))
    shot_index = int(shot_index)
    background_name = background or BACKGROUND_NAMES[shot_index % len(BACKGROUND_NAMES)]
    if background_name not in assets.backgrounds:
        raise ValueError(f"Background sconosciuto: {background_name}")

    if destination is None:
        frame_dir = assets.root / "frames"
        destination_path = frame_dir / (
            f"shot-{shot_index:04d}-{background_name}-p{round(progress * 10000):04d}"
            f"-a{round(accent_value * 1000):03d}.png"
        )
        if destination_path.is_file():
            return destination_path
    else:
        destination_path = Path(destination)

    plate = _open_oriented(Path(assets.backgrounds[background_name]), "RGB")
    canvas = _moving_background(plate, progress, shot_index, accent_value)
    master = _open_oriented(assets.subject_sprite, "RGBA")

    phase = math.radians((shot_index * 97 + 17) % 360)
    entrance = 0.10 * math.exp(-5.2 * progress) * math.cos(progress * math.pi * 2.4)
    accent_pulse = accent_value * 0.075 * math.exp(-6.0 * progress)
    target_height = round(HEIGHT * (0.565 + entrance + accent_pulse))
    target_width = round(WIDTH * (0.76 + accent_value * 0.035))
    sprite = _fit_sprite(master, target_width, target_height)
    tilt = math.sin(phase + progress * math.pi * 1.65) * (0.48 + 0.26 * accent_value)
    sprite = _perspective_sprite(sprite, tilt)
    sprite = sprite.rotate(
        tilt * (3.2 + accent_value * 1.8),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    if sprite.width > WIDTH * 0.84 or sprite.height > HEIGHT * 0.68:
        sprite = _fit_sprite(sprite, round(WIDTH * 0.84), round(HEIGHT * 0.68))

    # Background moves far enough to sell depth; the master subject only
    # breathes around the visual centre and therefore remains recognisable.
    centre_x = WIDTH * 0.5 + math.sin(phase + progress * math.pi) * 14
    bottom = HEIGHT * 0.82 + math.cos(phase + progress * math.pi * 1.3) * 9
    x = round(centre_x - sprite.width / 2)
    y = round(bottom - sprite.height)
    canvas = Image.alpha_composite(
        canvas,
        _shadow_layer(sprite.width, sprite.height, centre_x, bottom, accent_value),
    )

    effects = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    travel_x = math.cos(phase) * (10 + accent_value * 24)
    travel_y = math.sin(phase) * (7 + accent_value * 13)
    # Duplicate trails and RGB-separated ghosts remain behind the clean
    # master, so identity and facial detail never disappear on an accent.
    for distance, opacity in ((1.85, 0.07 + accent_value * 0.12), (1.0, 0.10 + accent_value * 0.16)):
        ghost = _tinted(sprite, (126, 104, 255), opacity)
        effects.alpha_composite(
            ghost,
            dest=(round(x - travel_x * distance), round(y - travel_y * distance)),
        )
    split = 3 + round(accent_value * 10)
    effects.alpha_composite(_tinted(sprite, (255, 45, 86), 0.10 + accent_value * 0.17), dest=(x - split, y))
    effects.alpha_composite(_tinted(sprite, (38, 224, 255), 0.10 + accent_value * 0.17), dest=(x + split, y))
    # A rim is only a short impact cue.  A permanent thick cyan edge reads as
    # a bad matte, especially on black fur against the roulette background.
    effects.alpha_composite(_rim(sprite, 0.04 + accent_value * 0.12), dest=(x, y))
    canvas = Image.alpha_composite(canvas, effects)
    foreground = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    foreground.alpha_composite(sprite, dest=(x, y))
    canvas = Image.alpha_composite(canvas, foreground)

    _atomic_png(canvas.convert("RGBA"), destination_path)
    return destination_path


# A renderer can use either name without an adapter.
render_layered_frame = render_layered_board


__all__ = [
    "ASSET_VERSION",
    "BACKGROUND_NAMES",
    "HEIGHT",
    "LayeredAssets",
    "SOURCE_PROXY_MAX_SIDE",
    "WIDTH",
    "prepare_layered_assets",
    "render_layered_board",
    "render_layered_frame",
]
