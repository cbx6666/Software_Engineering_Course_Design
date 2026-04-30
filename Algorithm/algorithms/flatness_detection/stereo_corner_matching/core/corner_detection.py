"""Chessboard corner detection helpers."""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ...config import DEFAULT_CONFIG, CornerDetectionConfig
    from ...errors import CornerDetectionError
    from ...logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, CornerDetectionConfig
    from errors import CornerDetectionError
    from logging_utils import get_logger


logger = get_logger("corners.detection")


def _as_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    out = img.astype(np.float32, copy=False)
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def enhance_chessboard_region(img: np.ndarray, config: CornerDetectionConfig = None) -> np.ndarray:
    """Enhance chessboard contrast with CLAHE."""
    config = config or DEFAULT_CONFIG.corner_detection
    img = _as_gray_uint8(img)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    return clahe.apply(img)


def _preprocess_image(img: np.ndarray, mode: str, config: CornerDetectionConfig) -> np.ndarray:
    img = _as_gray_uint8(img)
    if mode == "original":
        return img
    if mode == "clahe":
        return enhance_chessboard_region(img, config=config)
    if mode == "blur":
        k = max(3, config.blur_kernel | 1)
        return cv2.GaussianBlur(img, (k, k), 0)
    if mode == "threshold":
        block = max(3, config.threshold_block_size | 1)
        return cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            config.threshold_c,
        )
    if mode == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)
    return img


def _apply_mask(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return img
    mask = _as_gray_uint8(mask)
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(img, mask)


def _normalize_corners(corners: np.ndarray) -> Optional[np.ndarray]:
    if corners is None:
        return None
    corners = np.asarray(corners, dtype=np.float32)
    if corners.size == 0:
        return None
    if corners.ndim == 2:
        corners = corners.reshape(-1, 1, 2)
    if corners.ndim != 3 or corners.shape[2] != 2:
        return None
    return corners


def _opencv_result(result) -> Tuple[bool, Optional[np.ndarray]]:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return bool(result[0]), _normalize_corners(result[1])
        return False, None
    return result is not None, _normalize_corners(result)


def _sb_flag_list() -> List[int]:
    flags = [cv2.CALIB_CB_NORMALIZE_IMAGE]
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags.append(cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags.append(cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE)
    return flags


def _classic_flags(allow_partial: bool) -> int:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    partial_flag = getattr(cv2, "CALIB_CB_PARTIAL_OK", 0)
    if allow_partial and partial_flag:
        flags |= partial_flag
    return flags


def _required_count(pattern_size: Tuple[int, int], allow_partial: bool, config: CornerDetectionConfig) -> int:
    expected = pattern_size[0] * pattern_size[1]
    if not allow_partial:
        return expected
    return max(3, int(np.ceil(expected * config.min_corner_ratio)))


def _nearest_distances(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.array([], dtype=np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def _validate_full_grid_orientation(
    pts: np.ndarray,
    grid_shape: Tuple[int, int],
    config: CornerDetectionConfig,
) -> Tuple[bool, str, Dict[str, float]]:
    rows, cols = grid_shape
    grid = pts.reshape(rows, cols, 2)
    row_steps = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    col_steps = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    all_steps = np.concatenate([row_steps.ravel(), col_steps.ravel()])
    median_step = float(np.median(all_steps))
    metrics: Dict[str, float] = {
        "grid_rows": float(rows),
        "grid_cols": float(cols),
        "grid_median_step": median_step,
    }
    if median_step < config.min_spacing:
        return False, "完整棋盘格角点间距异常偏小", metrics
    if float(np.max(all_steps)) > median_step * config.max_spacing_ratio:
        return False, "完整棋盘格角点间距存在异常跳变", metrics

    row_axis = np.median(np.diff(grid, axis=1).reshape(-1, 2), axis=0)
    col_axis = np.median(np.diff(grid, axis=0).reshape(-1, 2), axis=0)
    if np.linalg.norm(row_axis) > 1e-6 and np.linalg.norm(col_axis) > 1e-6:
        row_axis = row_axis / np.linalg.norm(row_axis)
        col_axis = col_axis / np.linalg.norm(col_axis)
        row_proj = grid @ row_axis
        col_proj = grid @ col_axis
        row_positive = np.mean(np.diff(row_proj, axis=1) > -0.25 * median_step)
        col_positive = np.mean(np.diff(col_proj, axis=0) > -0.25 * median_step)
        metrics["row_monotonic_ratio"] = float(row_positive)
        metrics["col_monotonic_ratio"] = float(col_positive)
        if row_positive < 0.9 or col_positive < 0.9:
            return False, "角点行列排序单调性不足", metrics

    return True, "ok", metrics


def validate_corner_quality(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    pattern_size: Tuple[int, int],
    allow_partial: bool = True,
    config: CornerDetectionConfig = None,
) -> Tuple[bool, str, Dict[str, float]]:
    """Validate corner count, coverage, duplicates, and full-grid monotonicity."""
    config = config or DEFAULT_CONFIG.corner_detection
    corners = _normalize_corners(corners)
    if corners is None:
        return False, "角点数组为空或格式无效", {}

    pts = corners.reshape(-1, 2)
    expected = pattern_size[0] * pattern_size[1]
    required = _required_count(pattern_size, allow_partial, config)
    count = len(pts)
    metrics: Dict[str, float] = {
        "count": float(count),
        "expected_count": float(expected),
        "required_count": float(required),
    }

    if count < required:
        return False, f"角点数量不足：检测到 {count} 个，至少需要 {required} 个", metrics

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    h, w = image_shape[:2]
    coverage = max((x_max - x_min) * (y_max - y_min), 0.0) / max(float(h * w), 1.0)
    metrics["coverage"] = float(coverage)
    if count < expected and coverage < config.min_area_coverage:
        return False, f"角点覆盖区域过小：coverage={coverage:.4f}", metrics
    if count == expected and coverage < config.min_area_coverage:
        logger.warning(
            "full chessboard corners detected with small image coverage %.4f; continuing with geometry checks",
            coverage,
        )

    nearest = _nearest_distances(pts)
    if nearest.size:
        metrics["nearest_min"] = float(np.min(nearest))
        metrics["nearest_median"] = float(np.median(nearest))
        if np.min(nearest) < config.min_duplicate_distance:
            return False, "存在明显重复角点", metrics
        if np.median(nearest) < config.min_spacing:
            return False, "角点平均间距异常偏小", metrics

    if count == expected:
        rows, cols = pattern_size
        orientations = [(rows, cols)]
        if rows != cols:
            orientations.append((cols, rows))

        failures: List[str] = []
        for grid_shape in orientations:
            valid, reason, grid_metrics = _validate_full_grid_orientation(pts, grid_shape, config)
            if valid:
                metrics.update(grid_metrics)
                return True, "ok", metrics
            failures.append(reason)
        return False, failures[-1] if failures else "角点行列排序单调性不足", metrics

    return True, "ok", metrics


def _try_sb(img: np.ndarray, pattern_size: Tuple[int, int], flags: int) -> Tuple[bool, Optional[np.ndarray]]:
    if not hasattr(cv2, "findChessboardCornersSB"):
        return False, None
    try:
        return _opencv_result(cv2.findChessboardCornersSB(img, pattern_size, flags=flags))
    except cv2.error as exc:
        logger.debug("findChessboardCornersSB failed: %s", exc)
        return False, None


def _try_classic(img: np.ndarray, pattern_size: Tuple[int, int], flags: int) -> Tuple[bool, Optional[np.ndarray]]:
    try:
        return _opencv_result(cv2.findChessboardCorners(img, pattern_size, flags=flags))
    except cv2.error as exc:
        logger.debug("findChessboardCorners failed: %s", exc)
        return False, None


def find_chessboard_corners(
    img: np.ndarray,
    pattern_size: Tuple[int, int],
    allow_partial: bool = True,
    enhance: bool = False,
    mask: Optional[np.ndarray] = None,
    config: CornerDetectionConfig = None,
    raise_on_failure: bool = False,
) -> Optional[np.ndarray]:
    """
    Detect chessboard corners with SB, classic, and preprocessing fallbacks.

    The return shape remains compatible with the old code: (N, 1, 2), or None
    when detection fails and raise_on_failure is False.
    """
    config = config or DEFAULT_CONFIG.corner_detection
    base = enhance_chessboard_region(img, config=config) if enhance else _as_gray_uint8(img)
    preprocess_modes = config.preprocess_modes
    failures: List[str] = []

    for mode in preprocess_modes:
        processed = _apply_mask(_preprocess_image(base, mode, config), mask)
        for scale in config.scales:
            if scale == 1.0:
                scaled = processed
            else:
                scaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            attempts: List[Tuple[str, int]] = [("sb", flags) for flags in _sb_flag_list()]
            attempts.append(("classic", _classic_flags(allow_partial)))
            for method, flags in attempts:
                ok, corners = (
                    _try_sb(scaled, pattern_size, flags)
                    if method == "sb"
                    else _try_classic(scaled, pattern_size, flags)
                )
                if not ok or corners is None:
                    continue

                if scale != 1.0:
                    corners = corners / scale

                valid, reason, metrics = validate_corner_quality(
                    corners,
                    processed.shape,
                    pattern_size,
                    allow_partial=allow_partial,
                    config=config,
                )
                if valid:
                    logger.info(
                        "corner detection succeeded: method=%s mode=%s scale=%.2f count=%d/%d",
                        method,
                        mode,
                        scale,
                        len(corners),
                        pattern_size[0] * pattern_size[1],
                    )
                    return corners

                failures.append(f"{method}/{mode}/scale={scale}: {reason}")

    message = "角点检测失败：" + ("; ".join(failures[-5:]) if failures else "所有策略均未返回角点")
    logger.error(message)
    if raise_on_failure:
        raise CornerDetectionError(message)
    return None


def refine_corners(
    img: np.ndarray,
    corners: np.ndarray,
    config: CornerDetectionConfig = None,
) -> np.ndarray:
    """Refine corners with cornerSubPix, falling back to original corners on failure."""
    config = config or DEFAULT_CONFIG.corner_detection
    corners = _normalize_corners(corners)
    if corners is None or len(corners) == 0:
        return corners

    gray = _as_gray_uint8(img)
    corners = corners.astype(np.float32, copy=True)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.subpix_max_iter,
        config.subpix_eps,
    )
    try:
        return cv2.cornerSubPix(
            gray,
            corners,
            config.subpix_window,
            config.subpix_zero_zone,
            criteria,
        )
    except cv2.error as exc:
        logger.warning("cornerSubPix failed, using original corners: %s", exc)
        return corners


def draw_corners_visualization(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(_as_gray_uint8(img), cv2.COLOR_GRAY2BGR)
    corners = _normalize_corners(corners)
    if corners is None:
        return vis
    for idx, pt in enumerate(corners.reshape(-1, 2), start=1):
        x, y = np.round(pt).astype(int)
        cv2.circle(vis, (x, y), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
    return vis
