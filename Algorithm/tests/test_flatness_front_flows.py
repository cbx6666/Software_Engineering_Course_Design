import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLATNESS_DIR = ROOT / "algorithms" / "flatness_detection"
if str(FLATNESS_DIR) not in sys.path:
    sys.path.insert(0, str(FLATNESS_DIR))


from algorithms.flatness_detection.config import (  # noqa: E402
    CornerDetectionConfig,
    CornerMatchingConfig,
    FlatnessConfig,
    ProjectionDiffConfig,
)
from algorithms.flatness_detection.errors import CornerMatchingError  # noqa: E402
from algorithms.flatness_detection.projector_reflection_diff.core.alignment import ecc_register  # noqa: E402
from algorithms.flatness_detection.projector_reflection_diff.core.brightness_matching import (  # noqa: E402
    build_fit_mask,
    channelwise_compensated_diff,
    robust_fit_beta_gamma,
)
from algorithms.flatness_detection.projector_reflection_diff.core.chessboard_processing import (  # noqa: E402
    build_chess_mask_from_proj,
)
from algorithms.flatness_detection.projector_reflection_diff.main import process_projection_diff  # noqa: E402
from algorithms.flatness_detection.stereo_corner_matching.core.corner_detection import (  # noqa: E402
    find_chessboard_corners,
    refine_corners,
    validate_corner_quality,
)
from algorithms.flatness_detection.stereo_corner_matching.core.corner_matching import (  # noqa: E402
    match_by_relative_coordinates,
)


def make_bgr_gradient(h=80, w=100):
    x = np.tile(np.linspace(20, 200, w, dtype=np.uint8), (h, 1))
    y = np.tile(np.linspace(10, 120, h, dtype=np.uint8)[:, None], (1, w))
    return cv2.merge([x, y, ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)])


def make_checkerboard(pattern_size=(7, 10), square=28, margin=20, contrast=220):
    cols, rows = pattern_size
    board_cols = cols + 1
    board_rows = rows + 1
    h = board_rows * square + margin * 2
    w = board_cols * square + margin * 2
    img = np.full((h, w), 128, dtype=np.uint8)
    for r in range(board_rows):
        for c in range(board_cols):
            value = contrast if (r + c) % 2 == 0 else 255 - contrast
            y1 = margin + r * square
            x1 = margin + c * square
            img[y1:y1 + square, x1:x1 + square] = value
    return img


def make_corner_grid(pattern_size=(7, 10), offset=(40.0, 30.0), spacing=(12.0, 9.0), disparity=6.0):
    rows, cols = pattern_size
    left = []
    right = []
    for r in range(rows):
        for c in range(cols):
            x = offset[0] + c * spacing[0] + r * 0.7
            y = offset[1] + r * spacing[1] + c * 0.15
            left.append([x, y])
            right.append([x - disparity, y + 0.3])
    return (
        np.asarray(left, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(right, dtype=np.float32).reshape(-1, 1, 2),
    )


class ProjectionDiffTests(unittest.TestCase):
    def test_ecc_fallback_returns_image_when_opencv_fails(self):
        moving = make_bgr_gradient()
        fixed = make_bgr_gradient()
        with mock.patch("cv2.findTransformECC", side_effect=cv2.error("forced failure")):
            aligned = ecc_register(moving, fixed, config=ProjectionDiffConfig())
        self.assertEqual(aligned.shape, fixed.shape)
        self.assertTrue(np.array_equal(aligned, moving))

    def test_fit_mask_survives_size_and_brightness_changes(self):
        env = make_bgr_gradient(60, 90)
        mix = np.clip(env.astype(np.float32) * 1.18 + 12, 0, 255).astype(np.uint8)
        mix[20:35, 25:45] = 255
        mask = build_fit_mask(env, mix, config=ProjectionDiffConfig())
        self.assertEqual(mask.shape, env.shape[:2])
        self.assertGreater(np.count_nonzero(mask), 0)
        beta, gamma = robust_fit_beta_gamma(env[:, :, 0], mix[:, :, 0], mask, config=ProjectionDiffConfig())
        self.assertGreater(beta, 0.7)
        diff = channelwise_compensated_diff(env, mix, mask, config=ProjectionDiffConfig())
        self.assertEqual(diff.dtype, np.uint8)

    def test_chess_mask_filters_small_components(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (75, 75), 180, -1)
        cv2.circle(img, (5, 5), 2, 255, -1)
        config = ProjectionDiffConfig(chess_min_area=100, chess_keep_regions=1)
        mask = build_chess_mask_from_proj(img, config=config)
        self.assertEqual(mask.shape, img.shape)
        self.assertGreater(np.count_nonzero(mask[20:80, 20:80]), 0)

    def test_process_projection_diff_handles_different_input_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_bgr_gradient(80, 100)
            mix = make_bgr_gradient(90, 120)
            cv2.rectangle(mix, (30, 25), (90, 70), (240, 240, 240), -1)
            env_path = os.path.join(tmp, "env.png")
            mix_path = os.path.join(tmp, "mix.png")
            cv2.imwrite(env_path, env)
            cv2.imwrite(mix_path, mix)
            detect_img, mask = process_projection_diff(env_path, mix_path, tmp, config=ProjectionDiffConfig())
            self.assertEqual(detect_img.shape, mask.shape)
            self.assertEqual(detect_img.shape, mix.shape[:2])


class CornerDetectionTests(unittest.TestCase):
    def test_detects_normal_checkerboard(self):
        img = make_checkerboard()
        config = CornerDetectionConfig(min_corner_ratio=0.5, min_area_coverage=0.005)
        corners = find_chessboard_corners(img, config.chessboard_size, config=config)
        self.assertIsNotNone(corners)
        self.assertGreaterEqual(len(corners), int(70 * config.min_corner_ratio))

    def test_detects_low_contrast_checkerboard_with_fallbacks(self):
        img = make_checkerboard(contrast=150)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        config = CornerDetectionConfig(min_corner_ratio=0.5, min_area_coverage=0.005)
        corners = find_chessboard_corners(img, config.chessboard_size, config=config)
        self.assertIsNotNone(corners)

    def test_refine_corners_falls_back_on_invalid_image_format(self):
        img = make_checkerboard()
        corners = np.array([[[10.0, 10.0]], [[20.0, 20.0]]], dtype=np.float32)
        with mock.patch("cv2.cornerSubPix", side_effect=cv2.error("forced")):
            refined = refine_corners(img, corners, config=CornerDetectionConfig())
        self.assertEqual(refined.shape, corners.shape)

    def test_full_corner_set_is_not_rejected_by_small_image_coverage(self):
        corners, _ = make_corner_grid(offset=(20.0, 20.0), spacing=(10.0, 7.0))
        valid, reason, metrics = validate_corner_quality(
            corners,
            image_shape=(640, 1200),
            pattern_size=(7, 10),
            allow_partial=True,
            config=CornerDetectionConfig(min_area_coverage=0.01),
        )
        self.assertTrue(valid, reason)
        self.assertLess(metrics["coverage"], 0.01)


class CornerMatchingTests(unittest.TestCase):
    def test_matches_complete_corner_grid_and_reports_quality(self):
        left, right = make_corner_grid()
        matched_left, matched_right, quality = match_by_relative_coordinates(
            left,
            right,
            (7, 10),
            config=CornerMatchingConfig(),
            return_quality=True,
        )
        self.assertEqual(len(matched_left), 70)
        self.assertEqual(len(matched_right), 70)
        self.assertEqual(quality["matched_count"], 70)
        self.assertGreater(quality["disparity_mean"], 0)

    def test_matches_with_missing_corner_subset(self):
        left, right = make_corner_grid()
        keep = np.ones(len(left), dtype=bool)
        keep[[3, 17, 33, 51]] = False
        matched_left, matched_right, quality = match_by_relative_coordinates(
            left[keep],
            right[keep],
            (7, 10),
            config=CornerMatchingConfig(min_match_ratio=0.5),
            return_quality=True,
        )
        self.assertEqual(len(matched_left), 66)
        self.assertEqual(quality["outlier_count"], 0)

    def test_disparity_outlier_is_filtered(self):
        left, right = make_corner_grid()
        right[10, 0, 0] = left[10, 0, 0] - 10.0
        matched_left, matched_right, quality = match_by_relative_coordinates(
            left,
            right,
            (7, 10),
            config=CornerMatchingConfig(max_outlier_ratio=0.35),
            return_quality=True,
        )
        self.assertEqual(len(matched_left), 69)
        self.assertEqual(quality["outlier_count"], 1)

    def test_too_few_matches_raise_clear_error(self):
        left, right = make_corner_grid()
        with self.assertRaises(CornerMatchingError):
            match_by_relative_coordinates(
                left[:5],
                right[:5],
                (7, 10),
                config=CornerMatchingConfig(min_match_ratio=0.5),
            )

    def test_high_geometry_outlier_ratio_keeps_matches_in_compat_mode(self):
        left, right = make_corner_grid()
        right[:, 0, 1] += 100.0
        matched_left, matched_right, quality = match_by_relative_coordinates(
            left,
            right,
            (7, 10),
            config=CornerMatchingConfig(strict_geometry=False),
            return_quality=True,
        )
        self.assertEqual(len(matched_left), 70)
        self.assertEqual(len(matched_right), 70)
        self.assertEqual(quality["outlier_count"], 70)

    def test_high_geometry_outlier_ratio_can_be_strict_failure(self):
        left, right = make_corner_grid()
        right[:, 0, 1] += 100.0
        with self.assertRaises(CornerMatchingError):
            match_by_relative_coordinates(
                left,
                right,
                (7, 10),
                config=CornerMatchingConfig(strict_geometry=True),
            )


class DetectMainCompatibilityTests(unittest.TestCase):
    def test_detect_main_keeps_output_contract_when_services_call_it(self):
        from algorithms.flatness_detection import detect

        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as result_dir:
            for name in ("left_env", "left_mix", "right_env", "right_mix"):
                cv2.imwrite(os.path.join(data_dir, f"{name}.png"), make_bgr_gradient(40, 50))

            left, right = make_corner_grid()
            with mock.patch.object(detect, "process_projection_diff", return_value=(np.zeros((40, 50), np.uint8), np.ones((40, 50), np.uint8) * 255)), \
                 mock.patch.object(detect, "detect_corners", side_effect=[left, right]), \
                 mock.patch.object(detect, "process_stereo_matches", return_value={
                     "flatness_metrics": {"count": 70, "flatness_rms_mm": 0.0},
                     "projected_pts": np.zeros((70, 3)),
                     "projected_z": np.zeros(70),
                 }):
                detect.main(data_dir=data_dir, result_dir=result_dir, config=FlatnessConfig())

            with open(os.path.join(result_dir, "corners_left.json"), "r", encoding="utf-8") as f:
                saved_left = json.load(f)
            with open(os.path.join(result_dir, "corners_right.json"), "r", encoding="utf-8") as f:
                saved_right = json.load(f)

            self.assertEqual(len(saved_left), 70)
            self.assertEqual(len(saved_right), 70)
            self.assertTrue(os.path.exists(os.path.join(result_dir, "match_quality.json")))
            self.assertTrue(os.path.exists(os.path.join(result_dir, "flatness_metrics.json")))


if __name__ == "__main__":
    unittest.main()
