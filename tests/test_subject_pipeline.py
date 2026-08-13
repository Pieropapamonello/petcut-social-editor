import unittest

import cv2
import numpy as np

from subject_pipeline import (
    SubjectCandidate,
    decontaminate_edges,
    describe_identity,
    identity_similarity,
    matte_diagnostics,
    matte_gate,
    refine_alpha,
    select_identity_cohort,
)


def synthetic_subject(
    colour=(175, 72, 36),
    *,
    background=(24, 135, 54),
    shift=0,
    shape=(180, 240),
):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    body = ((xx - width * 0.50 - shift) / (width * 0.23)) ** 2 + ((yy - height * 0.56) / (height * 0.34)) ** 2 <= 1
    head = (xx - width * 0.50 - shift) ** 2 + (yy - height * 0.25) ** 2 <= (height * 0.13) ** 2
    left_ear = ((xx - width * 0.42 - shift) / 14) ** 2 + ((yy - height * 0.12) / 25) ** 2 <= 1
    right_ear = ((xx - width * 0.58 - shift) / 14) ** 2 + ((yy - height * 0.12) / 25) ** 2 <= 1
    mask = body | head | left_ear | right_ear
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[:] = background
    rgb[mask] = colour
    # Stable texture makes the identity hashes meaningful without a real photo.
    stripes = mask & (((xx + yy) // 13) % 5 == 0)
    rgb[stripes] = np.maximum(0, np.asarray(colour) - 28)
    alpha = (mask.astype(np.uint8) * 255)
    return rgb, alpha


class SubjectPipelineTests(unittest.TestCase):
    def test_descriptor_ignores_background_and_separates_identity(self):
        first_rgb, first_alpha = synthetic_subject(background=(10, 180, 40), shift=-2)
        same_rgb, same_alpha = synthetic_subject(background=(180, 30, 180), shift=3)
        other_rgb, other_alpha = synthetic_subject(colour=(28, 70, 190), background=(10, 180, 40))
        first = describe_identity(first_rgb, first_alpha)
        same = describe_identity(same_rgb, same_alpha)
        other = describe_identity(other_rgb, other_alpha)
        self.assertGreater(identity_similarity(first, same), 0.75)
        self.assertGreater(identity_similarity(first, same), identity_similarity(first, other) + 0.14)
        self.assertLessEqual(first.hsv_hist.nbytes, 2048)

    def test_matte_gate_rejects_rectangles_fragments_and_holes(self):
        _, clean = synthetic_subject()
        self.assertTrue(matte_gate(clean), matte_diagnostics(clean))

        rectangle = np.full(clean.shape, 255, np.uint8)
        self.assertFalse(matte_gate(rectangle))
        self.assertIn("background-rectangle", matte_diagnostics(rectangle).reasons)

        fragments = np.zeros_like(clean)
        for x, y in ((20, 25), (100, 30), (170, 45), (45, 130), (145, 135)):
            cv2.circle(fragments, (x, y), 13, 255, -1)
        self.assertFalse(matte_gate(fragments))

        perforated = clean.copy()
        yy, xx = np.mgrid[: clean.shape[0], : clean.shape[1]]
        for x, y in ((93, 61), (120, 72), (105, 101), (130, 122), (94, 137)):
            perforated[(xx - x) ** 2 + (yy - y) ** 2 <= 9**2] = 0
        self.assertFalse(matte_gate(perforated))
        self.assertIn("perforated", matte_diagnostics(perforated).reasons)

    def test_cohort_keeps_one_identity_and_rejects_bad_matte(self):
        first_rgb, first_alpha = synthetic_subject(shift=-2, background=(20, 140, 35))
        second_rgb, second_alpha = synthetic_subject(shift=3, background=(165, 25, 170))
        third_rgb, third_alpha = synthetic_subject(shift=0, background=(30, 40, 170))
        other_rgb, other_alpha = synthetic_subject(colour=(30, 70, 195), background=(20, 140, 35))
        broken_rgb, _ = synthetic_subject()
        broken_alpha = np.full(first_alpha.shape, 255, np.uint8)
        candidates = [
            SubjectCandidate(first_rgb, first_alpha, source_index=0, class_id="dog", candidate_id="hero-a"),
            SubjectCandidate(second_rgb, second_alpha, source_index=0, class_id="dog", candidate_id="hero-b"),
            SubjectCandidate(third_rgb, third_alpha, source_index=1, class_id="dog", candidate_id="hero-c"),
            SubjectCandidate(other_rgb, other_alpha, source_index=2, class_id="dog", candidate_id="other-dog"),
            SubjectCandidate(broken_rgb, broken_alpha, source_index=3, class_id="dog", candidate_id="room"),
        ]
        result = select_identity_cohort(candidates, minimum_similarity=0.66)
        self.assertIn(result.anchor_index, (0, 1, 2))
        self.assertEqual(set(result.member_indices), {0, 1, 2})
        self.assertEqual(set(result.rejected_indices), {3, 4})

    def test_detector_class_is_a_hard_cohort_constraint(self):
        rgb, alpha = synthetic_subject()
        candidates = [
            SubjectCandidate(rgb, alpha, source_index=0, class_id="dog"),
            SubjectCandidate(rgb.copy(), alpha.copy(), source_index=1, class_id="cat"),
        ]
        result = select_identity_cohort(candidates)
        self.assertEqual(len(result.member_indices), 1)

    def test_refine_alpha_removes_remote_island_and_tracks_core(self):
        rgb, truth = synthetic_subject(background=(20, 180, 50))
        coarse = cv2.dilate(truth, np.ones((9, 9), np.uint8), iterations=1)
        cv2.rectangle(coarse, (4, 7), (42, 43), 255, -1)
        # Add a background-coloured bite and a weak halo around the real subject.
        cv2.circle(coarse, (120, 103), 8, 0, -1)
        coarse = cv2.GaussianBlur(coarse, (5, 5), 0)
        refined = refine_alpha(rgb, coarse, bbox=(58, 17, 126, 157))
        self.assertEqual(refined.dtype, np.uint8)
        self.assertEqual(refined.shape, truth.shape)
        self.assertLess(np.mean(refined[7:43, 4:42]), 12)
        predicted = refined >= 128
        target = truth >= 128
        intersection = np.count_nonzero(predicted & target)
        union = np.count_nonzero(predicted | target)
        self.assertGreater(intersection / union, 0.88)
        self.assertTrue(matte_gate(refined), matte_diagnostics(refined))

    def test_refine_alpha_can_recover_a_clipped_connected_limb(self):
        rgb, truth = synthetic_subject(background=(20, 180, 50))
        # Extend the same-colour foreground just beyond the coarse mask, like
        # a paw clipped by a low-resolution detector prototype.
        cv2.ellipse(truth, (184, 122), (22, 9), 0, 0, 360, 255, -1)
        rgb[truth > 0] = (175, 72, 36)
        coarse = truth.copy()
        coarse[:, 174:] = 0
        refined = refine_alpha(rgb, coarse)
        recovered_region = refined[113:132, 176:204]
        # The refinement reclaims a meaningful connected portion without
        # inventing an opaque shape all the way to the recovery-band edge.
        self.assertGreater(float(np.mean(recovered_region)), 50.0)
        self.assertTrue(matte_gate(refined), matte_diagnostics(refined))

    def test_dark_neutral_subject_suppresses_warm_edge_spill(self):
        height, width = 200, 180
        rgb = np.full((height, width, 3), (220, 220, 220), np.uint8)
        coarse = np.zeros((height, width), np.uint8)
        cv2.ellipse(coarse, (95, 100), (45, 75), 0, 0, 360, 255, -1)
        rgb[coarse > 0] = (25, 27, 30)
        # A narrow red rug strip touches the silhouette, mirroring the real
        # black-dog failure without turning a brown animal into test data.
        cv2.rectangle(coarse, (45, 65), (53, 130), 255, -1)
        rgb[65:131, 45:54] = (135, 30, 45)
        refined = refine_alpha(rgb, coarse)
        self.assertLess(float(np.mean(refined[70:125, 46:52])), 220.0)
        self.assertGreater(float(np.mean(refined[60:145, 75:115])), 245.0)

    def test_edge_decontamination_reduces_green_fringe(self):
        rgb, alpha = synthetic_subject(colour=(180, 70, 35), background=(15, 205, 30))
        soft = cv2.GaussianBlur(alpha, (9, 9), 0)
        edge = (soft > 24) & (soft < 220)
        contaminated = rgb.copy()
        contaminated[edge] = (
            contaminated[edge].astype(np.float32) * 0.35 + np.array((15, 205, 30), np.float32) * 0.65
        ).astype(np.uint8)
        cleaned = decontaminate_edges(contaminated, soft)
        before_green_cast = np.mean(contaminated[edge, 1].astype(float) - contaminated[edge, 0].astype(float))
        after_green_cast = np.mean(cleaned[edge, 1].astype(float) - cleaned[edge, 0].astype(float))
        self.assertLess(after_green_cast, before_green_cast - 8)


if __name__ == "__main__":
    unittest.main()
