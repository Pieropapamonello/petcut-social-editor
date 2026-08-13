"""Small, model-free helpers for keeping one subject through an edit.

The production detector is deliberately not imported here.  This module works on
already detected RGB crops and alpha mattes, which makes identity selection
deterministic, inexpensive, and easy to test without downloading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MatteDiagnostics:
    bbox: BBox
    foreground_ratio: float
    bbox_fill_ratio: float
    main_component_ratio: float
    component_count: int
    hole_ratio: float
    border_contact: float
    edge_softness: float
    solidity: float
    aspect_ratio: float
    quality: float
    reasons: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.reasons and self.quality >= 0.42


@dataclass(frozen=True)
class IdentityDescriptor:
    """Compact representation of a masked subject (well below 2 KiB)."""

    hsv_hist: np.ndarray
    phash: int
    dhash: int
    shape_hash: int
    bbox: BBox
    area_ratio: float
    aspect_ratio: float
    bbox_fill_ratio: float
    quality: float


@dataclass(frozen=True)
class SubjectCandidate:
    rgb: np.ndarray
    alpha: np.ndarray
    source_index: int
    class_id: int | str | None = None
    confidence: float = 1.0
    bbox: BBox | None = None
    candidate_id: str | int | None = None


@dataclass(frozen=True)
class CohortSelection:
    anchor_index: int | None
    member_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    descriptors: tuple[IdentityDescriptor, ...]
    similarities: tuple[float, ...]
    score: float


def _as_alpha(alpha: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    values = np.asarray(alpha)
    if values.ndim == 3:
        values = values[..., -1]
    if values.ndim != 2:
        raise ValueError("alpha must be a two-dimensional mask")
    if shape is not None and values.shape != shape:
        raise ValueError("rgb and alpha dimensions do not match")
    values = values.astype(np.float32, copy=False)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _as_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    values = values[..., :3]
    if values.dtype != np.uint8:
        maximum = float(np.nanmax(values)) if values.size else 0.0
        scale = 255.0 if maximum <= 1.5 else 1.0
        values = np.nan_to_num(values, nan=0.0).clip(0, 255 / scale) * scale
        values = values.astype(np.uint8)
    return np.ascontiguousarray(values)


def _clip_bbox(bbox: BBox | None, width: int, height: int) -> BBox:
    if bbox is None:
        return (0, 0, width, height)
    x, y, box_width, box_height = (int(round(value)) for value in bbox)
    x = max(0, min(width, x))
    y = max(0, min(height, y))
    right = max(x, min(width, x + max(0, box_width)))
    bottom = max(y, min(height, y + max(0, box_height)))
    return (x, y, right - x, bottom - y)


def _mask_bbox(binary: np.ndarray) -> BBox:
    points = cv2.findNonZero(binary.astype(np.uint8))
    if points is None:
        return (0, 0, 0, 0)
    return tuple(int(value) for value in cv2.boundingRect(points))


def _bit_similarity(first: int, second: int, bits: int = 64) -> float:
    return 1.0 - ((int(first) ^ int(second)).bit_count() / bits)


def _bits(values: np.ndarray) -> int:
    output = 0
    for value in values.reshape(-1):
        output = (output << 1) | int(bool(value))
    return output


def _image_hashes(gray: np.ndarray, alpha: np.ndarray) -> tuple[int, int, int]:
    masked = gray.astype(np.float32) * alpha + 127.0 * (1.0 - alpha)
    small = cv2.resize(masked, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small)
    low = coefficients[:8, :8]
    threshold = float(np.median(low.reshape(-1)[1:]))
    phash = _bits(low > threshold)

    strip = cv2.resize(masked, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = _bits(strip[:, 1:] > strip[:, :-1])
    shape = cv2.resize(alpha, (9, 8), interpolation=cv2.INTER_AREA)
    shape_hash = _bits(shape[:, 1:] > shape[:, :-1])
    return phash, dhash, shape_hash


def matte_diagnostics(alpha: np.ndarray, bbox: BBox | None = None) -> MatteDiagnostics:
    """Measure whether a matte looks like one usable, non-rectangular subject."""

    matte = _as_alpha(alpha)
    height, width = matte.shape
    binary = (matte >= 0.5).astype(np.uint8)
    foreground = int(np.count_nonzero(binary))
    image_area = max(1, width * height)
    measured_bbox = _mask_bbox(binary)
    if bbox is not None:
        supplied = _clip_bbox(bbox, width, height)
        # Diagnostics should never ignore foreground outside a detector box.
        if supplied[2] and supplied[3]:
            measured_bbox = _mask_bbox(binary)

    if foreground == 0 or measured_bbox[2] == 0 or measured_bbox[3] == 0:
        return MatteDiagnostics(
            bbox=(0, 0, 0, 0),
            foreground_ratio=0.0,
            bbox_fill_ratio=0.0,
            main_component_ratio=0.0,
            component_count=0,
            hole_ratio=0.0,
            border_contact=0.0,
            edge_softness=0.0,
            solidity=0.0,
            aspect_ratio=0.0,
            quality=0.0,
            reasons=("empty",),
        )

    x, y, box_width, box_height = measured_bbox
    bbox_area = max(1, box_width * box_height)
    foreground_ratio = foreground / image_area
    bbox_fill = foreground / bbox_area
    aspect = box_width / max(1, box_height)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
    main_area = int(component_areas.max()) if component_areas.size else 0
    significant = int(np.count_nonzero(component_areas >= max(12, foreground * 0.006)))
    main_component_ratio = main_area / max(1, foreground)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None
    if main_contour is None:
        solidity = 0.0
    else:
        contour_area = float(cv2.contourArea(main_contour))
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(main_contour))))
        solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0))

    crop = binary[y : y + box_height, x : x + box_width]
    # Flood the inverse from an artificial one-pixel exterior.  Flooding the
    # tight crop directly is incorrect when the silhouette touches its bbox.
    inverse = np.pad((crop == 0).astype(np.uint8), 1, constant_values=1)
    flood_mask = np.zeros((inverse.shape[0] + 2, inverse.shape[1] + 2), np.uint8)
    flooded = inverse.copy()
    cv2.floodFill(flooded, flood_mask, (0, 0), 2)
    holes = (inverse[1:-1, 1:-1] == 1) & (flooded[1:-1, 1:-1] != 2)
    hole_ratio = float(np.count_nonzero(holes) / max(1, foreground))

    border_pixels = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    border_contact = float(np.count_nonzero(border_pixels) / max(1, border_pixels.size))
    soft = (matte > 0.04) & (matte < 0.96)
    edge_softness = float(np.count_nonzero(soft) / max(1, foreground))

    coverage_score = float(np.clip((foreground_ratio - 0.008) / 0.055, 0, 1))
    coverage_score *= float(np.clip((0.92 - foreground_ratio) / 0.18, 0, 1))
    fill_score = float(np.clip((0.97 - bbox_fill) / 0.25, 0, 1))
    fill_score = max(fill_score, float(np.clip((bbox_fill - 0.13) / 0.35, 0, 1)) * 0.72)
    aspect_score = float(np.clip((aspect - 0.16) / 0.24, 0, 1) * np.clip((5.2 - aspect) / 1.8, 0, 1))
    hole_score = float(np.clip(1.0 - hole_ratio / 0.10, 0, 1))
    border_score = float(np.clip(1.0 - border_contact / 0.48, 0, 1))
    quality = (
        0.25 * main_component_ratio
        + 0.18 * solidity
        + 0.16 * coverage_score
        + 0.15 * fill_score
        + 0.12 * hole_score
        + 0.08 * border_score
        + 0.06 * aspect_score
    )
    if bbox_fill > 0.965:
        quality *= 0.30
    if significant > 3:
        quality *= max(0.35, 1.0 - (significant - 3) * 0.11)
    quality = float(np.clip(quality, 0.0, 1.0))

    reasons: list[str] = []
    if not 0.008 <= foreground_ratio <= 0.92:
        reasons.append("implausible-area")
    if bbox_fill > 0.975:
        reasons.append("background-rectangle")
    if main_component_ratio < 0.76:
        reasons.append("fragmented")
    if hole_ratio > 0.105:
        reasons.append("perforated")
    if border_contact > 0.58:
        reasons.append("touches-frame")
    if not 0.16 <= aspect <= 5.2:
        reasons.append("implausible-aspect")
    if quality < 0.42:
        reasons.append("low-quality")

    return MatteDiagnostics(
        bbox=measured_bbox,
        foreground_ratio=float(foreground_ratio),
        bbox_fill_ratio=float(bbox_fill),
        main_component_ratio=float(main_component_ratio),
        component_count=significant,
        hole_ratio=hole_ratio,
        border_contact=border_contact,
        edge_softness=edge_softness,
        solidity=solidity,
        aspect_ratio=float(aspect),
        quality=quality,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def matte_gate(alpha_or_diagnostics: np.ndarray | MatteDiagnostics, minimum_quality: float = 0.42) -> bool:
    diagnostics = (
        alpha_or_diagnostics
        if isinstance(alpha_or_diagnostics, MatteDiagnostics)
        else matte_diagnostics(alpha_or_diagnostics)
    )
    return not diagnostics.reasons and diagnostics.quality >= minimum_quality


def describe_identity(rgb: np.ndarray, alpha: np.ndarray, bbox: BBox | None = None) -> IdentityDescriptor:
    """Build a background-independent colour, texture, and silhouette descriptor."""

    image = _as_rgb(rgb)
    matte = _as_alpha(alpha, image.shape[:2])
    diagnostics = matte_diagnostics(matte, bbox)
    x, y, width, height = diagnostics.bbox
    if width == 0 or height == 0:
        return IdentityDescriptor(
            hsv_hist=np.zeros(288, np.float32),
            phash=0,
            dhash=0,
            shape_hash=0,
            bbox=diagnostics.bbox,
            area_ratio=0.0,
            aspect_ratio=0.0,
            bbox_fill_ratio=0.0,
            quality=0.0,
        )

    crop_rgb = image[y : y + height, x : x + width]
    crop_alpha = matte[y : y + height, x : x + width]
    # Descriptors are evaluated at bounded resolution even for 4K input.
    scale = min(1.0, 160.0 / max(width, height))
    sample_width = max(8, round(width * scale))
    sample_height = max(8, round(height * scale))
    sample_rgb = cv2.resize(crop_rgb, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
    sample_alpha = cv2.resize(crop_alpha, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2HSV)
    mask = (sample_alpha >= 0.45).astype(np.uint8) * 255
    histogram = cv2.calcHist([hsv], [0, 1, 2], mask, [18, 4, 4], [0, 180, 0, 256, 0, 256])
    histogram = histogram.reshape(-1).astype(np.float32)
    total = float(histogram.sum())
    if total:
        histogram /= total
    gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
    phash, dhash, shape_hash = _image_hashes(gray, sample_alpha)
    return IdentityDescriptor(
        hsv_hist=histogram,
        phash=phash,
        dhash=dhash,
        shape_hash=shape_hash,
        bbox=diagnostics.bbox,
        area_ratio=diagnostics.foreground_ratio,
        aspect_ratio=diagnostics.aspect_ratio,
        bbox_fill_ratio=diagnostics.bbox_fill_ratio,
        quality=diagnostics.quality,
    )


def identity_similarity(first: IdentityDescriptor, second: IdentityDescriptor) -> float:
    """Return a conservative [0, 1] same-subject score."""

    first_hist = np.asarray(first.hsv_hist, dtype=np.float32)
    second_hist = np.asarray(second.hsv_hist, dtype=np.float32)
    if first_hist.shape != second_hist.shape or not first_hist.size:
        histogram_similarity = 0.0
    else:
        histogram_similarity = 1.0 - float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_BHATTACHARYYA))
    phash_similarity = _bit_similarity(first.phash, second.phash)
    dhash_similarity = _bit_similarity(first.dhash, second.dhash)
    shape_similarity = _bit_similarity(first.shape_hash, second.shape_hash)
    aspect_similarity = float(
        np.exp(-abs(np.log(max(0.05, first.aspect_ratio) / max(0.05, second.aspect_ratio))))
    )
    score = (
        0.48 * histogram_similarity
        + 0.19 * phash_similarity
        + 0.10 * dhash_similarity
        + 0.13 * shape_similarity
        + 0.10 * aspect_similarity
    )
    # A broken matte must not become a convincing identity match merely because
    # its average colour resembles the anchor.
    quality_factor = 0.72 + 0.28 * min(first.quality, second.quality)
    return float(np.clip(score * quality_factor, 0.0, 1.0))


def select_identity_cohort(
    candidates: Sequence[SubjectCandidate],
    *,
    minimum_similarity: float = 0.66,
    minimum_quality: float = 0.42,
) -> CohortSelection:
    """Choose a representative anchor and all reliable views of that identity.

    Detector class IDs are treated as a hard constraint.  Source indexes are
    used as a diversity bonus, not as identity evidence: several shots from one
    upload remain usable, while a coherent identity observed in multiple uploads
    beats an accidental same-colour detection.
    """

    if not candidates:
        return CohortSelection(None, (), (), (), (), 0.0)
    descriptors = tuple(describe_identity(item.rgb, item.alpha, item.bbox) for item in candidates)
    valid = [
        index
        for index, descriptor in enumerate(descriptors)
        if descriptor.quality >= minimum_quality and matte_gate(matte_diagnostics(candidates[index].alpha), minimum_quality)
    ]
    if not valid:
        return CohortSelection(None, (), tuple(range(len(candidates))), descriptors, (), 0.0)

    best_anchor: int | None = None
    best_members: tuple[int, ...] = ()
    best_similarities: tuple[float, ...] = ()
    best_score = -1.0
    for anchor in valid:
        anchor_candidate = candidates[anchor]
        similarities: list[tuple[int, float]] = []
        for index in valid:
            candidate = candidates[index]
            if (
                anchor_candidate.class_id is not None
                and candidate.class_id is not None
                and anchor_candidate.class_id != candidate.class_id
            ):
                continue
            similarity = 1.0 if index == anchor else identity_similarity(descriptors[anchor], descriptors[index])
            if similarity >= minimum_similarity or index == anchor:
                similarities.append((index, similarity))
        # Prevent a weak bridge from joining two identities: each non-anchor view
        # must also agree with the median of the provisional cohort.
        if len(similarities) > 2:
            provisional = tuple(index for index, _ in similarities)
            filtered: list[tuple[int, float]] = []
            for index, anchor_similarity in similarities:
                pair_scores = [
                    identity_similarity(descriptors[index], descriptors[other])
                    for other in provisional
                    if other != index
                ]
                consensus = float(np.median(pair_scores)) if pair_scores else 1.0
                if index == anchor or consensus >= minimum_similarity - 0.07:
                    filtered.append((index, anchor_similarity))
            similarities = filtered

        members = tuple(index for index, _ in similarities)
        scores = tuple(score for _, score in similarities)
        qualities = [descriptors[index].quality * float(np.clip(candidates[index].confidence, 0, 1)) for index in members]
        distinct_sources = len({candidates[index].source_index for index in members})
        support = sum(0.45 * quality + 0.55 * similarity for quality, similarity in zip(qualities, scores))
        cohort_score = support + 0.16 * max(0, distinct_sources - 1) + 0.08 * np.sqrt(len(members))
        # A representative with clean edges is preferred among otherwise equal clusters.
        cohort_score += 0.15 * descriptors[anchor].quality
        if cohort_score > best_score:
            best_anchor = anchor
            best_members = members
            best_similarities = scores
            best_score = float(cohort_score)

    rejected = tuple(index for index in range(len(candidates)) if index not in best_members)
    return CohortSelection(
        anchor_index=best_anchor,
        member_indices=best_members,
        rejected_indices=rejected,
        descriptors=descriptors,
        similarities=best_similarities,
        score=max(0.0, best_score),
    )


def _expanded_bbox(bbox: BBox, width: int, height: int, padding: float = 0.08) -> BBox:
    x, y, box_width, box_height = _clip_bbox(bbox, width, height)
    pad_x = max(3, round(box_width * padding))
    pad_y = max(3, round(box_height * padding))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + box_width + pad_x)
    bottom = min(height, y + box_height + pad_y)
    return left, top, right - left, bottom - top


def _anchor_component(binary: np.ndarray, core: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return binary.astype(np.uint8)
    best_label = 0
    best_score = -1.0
    anchor_x, anchor_y = anchor
    diagonal = max(1.0, np.hypot(binary.shape[1], binary.shape[0]))
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & core))
        center_x, center_y = centroids[label]
        distance = np.hypot(center_x - anchor_x, center_y - anchor_y) / diagonal
        score = 5.0 * overlap + area * max(0.0, 0.55 - distance)
        if score > best_score:
            best_score = score
            best_label = label
    return (labels == best_label).astype(np.uint8)


def refine_alpha(
    rgb: np.ndarray,
    coarse_alpha: np.ndarray,
    bbox: BBox | None = None,
    *,
    iterations: int = 3,
    feather_radius: float = 1.25,
) -> np.ndarray:
    """Refine one coarse matte with a bounded-memory GrabCut trimap.

    Only the expanded subject ROI is processed.  The selected component must be
    connected to the eroded foreground core (or closest to the detector centre),
    which removes remote people, furniture, and detector islands.
    """

    image = _as_rgb(rgb)
    coarse = _as_alpha(coarse_alpha, image.shape[:2])
    height, width = coarse.shape
    measured = _mask_bbox(coarse >= 0.42)
    target_bbox = _clip_bbox(bbox, width, height) if bbox is not None else measured
    if target_bbox[2] == 0 or target_bbox[3] == 0:
        return np.zeros((height, width), np.uint8)
    # Leave enough colour context for GrabCut to recover a paw, ear or hand
    # that the coarse low-resolution instance mask clipped at its boundary.
    roi_bbox = _expanded_bbox(target_bbox, width, height, 0.25)
    left, top, roi_width, roi_height = roi_bbox
    roi_rgb = image[top : top + roi_height, left : left + roi_width]
    roi_alpha = coarse[top : top + roi_height, left : left + roi_width]

    minimum_side = max(1, min(roi_width, roi_height))
    kernel_size = max(3, round(minimum_side * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    probable = (roi_alpha >= 0.28).astype(np.uint8)
    likely = (roi_alpha >= 0.56).astype(np.uint8)
    core = cv2.erode((roi_alpha >= 0.82).astype(np.uint8), kernel, iterations=2)
    dilated = cv2.dilate(probable, kernel, iterations=2)
    recovery_size = min(25, max(7, kernel_size * 3 - 2))
    if recovery_size % 2 == 0:
        recovery_size += 1
    recovery_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (recovery_size, recovery_size)
    )
    recovery_band = cv2.dilate(probable, recovery_kernel, iterations=3)
    if not np.any(core):
        core = cv2.erode(likely, np.ones((3, 3), np.uint8), iterations=1)

    trimap = np.full((roi_height, roi_width), cv2.GC_PR_BGD, np.uint8)
    # Pixels immediately outside the learned matte remain probable background,
    # not certain background. GrabCut can reclaim matching limbs there; only
    # the exterior of the wider recovery band is locked out.
    # Outside the ordinary coarse dilation stays certain background.  A much
    # wider recovery band is considered only when its colour matches the
    # confident subject core; leaving that whole band as probable background
    # allowed red rugs and floor tiles to be reclaimed with clipped paws.
    trimap[dilated == 0] = cv2.GC_BGD
    if np.any(core):
        lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        core_colours = lab[core > 0]
        colour_median = np.median(core_colours, axis=0)
        colour_mad = np.median(np.abs(core_colours - colour_median), axis=0) + 6.0
        colour_distance = np.sqrt(
            np.sum(((lab - colour_median) / colour_mad) ** 2, axis=2)
        )
        colour_recovery = (
            (recovery_band > 0)
            & (probable == 0)
            & (colour_distance < 2.15)
        )
        trimap[colour_recovery] = cv2.GC_PR_FGD
    trimap[probable > 0] = cv2.GC_PR_FGD
    trimap[core > 0] = cv2.GC_FGD
    # A hard ROI frame makes GrabCut stable when a coarse mask touches one side.
    trimap[[0, -1], :] = cv2.GC_BGD
    trimap[:, [0, -1]] = cv2.GC_BGD

    result = likely.copy()
    if np.any(core) and np.any(trimap == cv2.GC_BGD):
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                trimap,
                None,
                background_model,
                foreground_model,
                int(np.clip(iterations, 2, 3)),
                cv2.GC_INIT_WITH_MASK,
            )
            result = np.isin(trimap, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        except cv2.error:
            result = likely.copy()

    local_anchor = (
        int(np.clip(target_bbox[0] + target_bbox[2] / 2 - left, 0, roi_width - 1)),
        int(np.clip(target_bbox[1] + target_bbox[3] / 2 - top, 0, roi_height - 1)),
    )
    result = _anchor_component(result, core.astype(bool), local_anchor)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    sigma = float(np.clip(feather_radius, 0.6, 2.0))
    feathered = cv2.GaussianBlur(result.astype(np.float32), (0, 0), sigma)
    result_core = cv2.erode(result, np.ones((3, 3), np.uint8), iterations=1)
    edge_result = (result > 0) & (result_core == 0)
    feathered[edge_result] = np.maximum(
        feathered[edge_result], np.minimum(1.0, roi_alpha[edge_result] + 0.20)
    )
    # Recovered interior pixels need an opaque alpha even though their coarse
    # probability was zero; otherwise the recovered paw still looks amputated.
    feathered[result_core > 0] = np.maximum(feathered[result_core > 0], 0.94)

    # Matte-level colour decontamination: semi-transparent pixels statistically
    # indistinguishable from the local background are suppressed.
    ring = (cv2.dilate(result, np.ones((5, 5), np.uint8), iterations=2) > 0) & (result == 0)
    edge = (feathered > 0.02) & (feathered < 0.94)
    if np.count_nonzero(ring) >= 12 and np.any(edge):
        background_colour = np.median(roi_rgb[ring].astype(np.float32), axis=0)
        colour_distance = np.linalg.norm(roi_rgb.astype(np.float32) - background_colour, axis=2)
        suppression = np.clip((colour_distance - 5.0) / 24.0, 0.0, 1.0)
        feathered[edge] *= 0.35 + 0.65 * suppression[edge]

    # Dark neutral pets photographed on a red rug are a common failure case:
    # a narrow opaque strip of carpet can stay joined to black fur even after
    # GrabCut.  Activate this only when the confident subject core is neutral
    # (never for brown/orange animals or colourful clothing), preserve the
    # face corridor, and suppress warm pixels only near the silhouette edge.
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    confident_core = result_core > 0
    if np.count_nonzero(confident_core) >= 80:
        core_chroma = np.sqrt(
            (lab[:, :, 1][confident_core] - 128.0) ** 2
            + (lab[:, :, 2][confident_core] - 128.0) ** 2
        )
        core_lightness = lab[:, :, 0][confident_core]
        if (
            float(np.median(core_chroma)) < 13
            and float(np.median(core_lightness)) < 100
        ):
            distance = cv2.distanceTransform(result.astype(np.uint8), cv2.DIST_L2, 5)
            edge_limit = float(np.clip(min(roi_width, roi_height) * 0.045, 8, 18))
            hue, saturation, value = cv2.split(hsv)
            warm = (
                ((hue < 28) | (hue > 155))
                & (saturation > 52)
                & (value > 42)
                & (result > 0)
                & (distance < edge_limit)
            )
            local_x = target_bbox[0] - left
            local_y = target_bbox[1] - top
            local_width, local_height = target_bbox[2], target_bbox[3]
            yy, xx = np.mgrid[:roi_height, :roi_width]
            protect_face = (
                ((xx - (local_x + local_width * 0.50)) / max(2.0, local_width * 0.24)) ** 2
                + ((yy - (local_y + local_height * 0.42)) / max(2.0, local_height * 0.30)) ** 2
                < 1.0
            )
            warm &= ~protect_face
            spill = cv2.morphologyEx(
                warm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            spill = cv2.dilate(spill, np.ones((3, 3), np.uint8), iterations=1) > 0
            if np.any(spill):
                feathered[spill] *= np.clip(
                    (distance[spill] - 1.0) / max(4.0, edge_limit * 0.42), 0.0, 1.0
                )
                feathered = cv2.GaussianBlur(feathered, (0, 0), 0.55)

    output = np.zeros((height, width), np.uint8)
    output[top : top + roi_height, left : left + roi_width] = np.rint(feathered.clip(0, 1) * 255).astype(np.uint8)
    return output


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove a local background colour cast from semi-transparent edge pixels."""

    image = _as_rgb(rgb)
    matte = _as_alpha(alpha, image.shape[:2])
    binary = matte >= 0.5
    bbox = _mask_bbox(matte > 0.015)
    if bbox[2] == 0 or bbox[3] == 0:
        return image.copy()
    left, top, width, height = _expanded_bbox(bbox, image.shape[1], image.shape[0], 0.04)
    roi = image[top : top + height, left : left + width].astype(np.float32)
    local_alpha = matte[top : top + height, left : left + width]
    local_binary = binary[top : top + height, left : left + width].astype(np.uint8)
    background_weight = 1.0 - local_alpha
    foreground_weight = local_alpha
    sigma = max(1.2, min(width, height) * 0.018)
    background = cv2.GaussianBlur(roi * background_weight[..., None], (0, 0), sigma)
    background /= np.maximum(
        cv2.GaussianBlur(background_weight, (0, 0), sigma)[..., None], 0.035
    )
    interior = cv2.GaussianBlur(roi * foreground_weight[..., None], (0, 0), sigma)
    interior /= np.maximum(
        cv2.GaussianBlur(foreground_weight, (0, 0), sigma)[..., None], 0.035
    )

    edge = (local_alpha > 0.08) & (local_alpha < 0.94)
    safe_alpha = np.maximum(local_alpha[..., None], 0.28)
    unmixed = (roi - (1.0 - local_alpha[..., None]) * background) / safe_alpha
    unmixed = np.clip(unmixed, 0, 255)
    # Very low-alpha pixels use the nearby foreground estimate, preventing the
    # unmixing equation from amplifying compression noise.
    low_alpha = local_alpha < 0.32
    unmixed[low_alpha] = interior[low_alpha]
    strength = np.clip((0.94 - local_alpha) / 0.72, 0, 0.78)[..., None]
    cleaned = np.where(edge[..., None], roi * (1.0 - strength) + unmixed * strength, roi)

    # Coarse instance masks sometimes leave one fully opaque row of carpet or
    # wall immediately inside the silhouette.  It is invisible to the soft
    # alpha test above, yet becomes a coloured fringe on black backgrounds.
    # Pull only this 1–2 px inner boundary toward confident foreground colour;
    # the alpha itself is preserved so fine fur is not shaved away.
    eroded = cv2.erode(local_binary, np.ones((3, 3), np.uint8), iterations=2)
    inner_boundary = (local_binary > 0) & (eroded == 0)
    usable_interior = (local_alpha >= 0.92).astype(np.float32) * eroded.astype(np.float32)
    interior_weight = cv2.GaussianBlur(usable_interior, (0, 0), max(1.0, sigma * 0.72))
    interior_colour = cv2.GaussianBlur(
        roi * usable_interior[..., None], (0, 0), max(1.0, sigma * 0.72)
    ) / np.maximum(interior_weight[..., None], 0.025)
    reliable_boundary = inner_boundary & (interior_weight > 0.04)
    cleaned[reliable_boundary] = (
        cleaned[reliable_boundary] * 0.38 + interior_colour[reliable_boundary] * 0.62
    )

    output = image.copy()
    output[top : top + height, left : left + width] = np.rint(cleaned.clip(0, 255)).astype(np.uint8)
    return output


def refine_cutout(
    rgb: np.ndarray,
    coarse_alpha: np.ndarray,
    bbox: BBox | None = None,
    *,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, MatteDiagnostics]:
    """Convenience API returning decontaminated RGB, alpha, and its gate data."""

    alpha = refine_alpha(rgb, coarse_alpha, bbox, iterations=iterations)
    clean_rgb = decontaminate_edges(rgb, alpha)
    return clean_rgb, alpha, matte_diagnostics(alpha, bbox)


__all__ = [
    "BBox",
    "CohortSelection",
    "IdentityDescriptor",
    "MatteDiagnostics",
    "SubjectCandidate",
    "decontaminate_edges",
    "describe_identity",
    "identity_similarity",
    "matte_diagnostics",
    "matte_gate",
    "refine_alpha",
    "refine_cutout",
    "select_identity_cohort",
]
