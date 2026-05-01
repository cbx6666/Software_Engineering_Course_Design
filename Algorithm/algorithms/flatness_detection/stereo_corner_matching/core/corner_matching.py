"""左右棋盘格角点匹配。"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from ...config import DEFAULT_CONFIG, CornerMatchingConfig
    from ...errors import CornerMatchingError
    from ...logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, CornerMatchingConfig
    from errors import CornerMatchingError
    from logging_utils import get_logger


logger = get_logger("corners.matching")


def _empty_grid(pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """创建带 NaN 的空网格和有效性标记。"""
    rows, cols = pattern_size
    total_expected = rows * cols
    return (
        np.full((total_expected, 1, 2), np.nan, dtype=np.float32),
        np.zeros(total_expected, dtype=bool),
    )


def _estimate_grid_axes(pts: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """根据点集整体分布估计棋盘格的行列方向。"""
    centered = pts - np.mean(pts, axis=0)
    if len(pts) < 2 or np.linalg.norm(centered) < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    primary = vh[0].astype(np.float32)
    secondary = vh[1].astype(np.float32) if vh.shape[0] > 1 else np.array([-primary[1], primary[0]], dtype=np.float32)

    rows, cols = pattern_size
    if cols >= rows:
        row_axis, col_axis = primary, secondary
    else:
        row_axis, col_axis = secondary, primary

    if abs(row_axis[0]) + abs(row_axis[1]) > 0 and row_axis[0] < 0:
        row_axis = -row_axis
    if abs(col_axis[0]) + abs(col_axis[1]) > 0 and col_axis[1] < 0:
        col_axis = -col_axis

    row_axis = row_axis / max(np.linalg.norm(row_axis), 1e-6)
    col_axis = col_axis - np.dot(col_axis, row_axis) * row_axis
    col_axis = col_axis / max(np.linalg.norm(col_axis), 1e-6)
    return row_axis, col_axis


def _cluster_axis(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """在一维投影上聚类，估计行或列的中心位置。"""
    values = values.astype(np.float32, copy=False)
    if n_clusters <= 1 or np.max(values) - np.min(values) < 1e-6:
        return np.full(n_clusters, float(np.mean(values)) if values.size else 0.0, dtype=np.float32)

    centers = np.linspace(float(np.min(values)), float(np.max(values)), n_clusters, dtype=np.float32)
    for _ in range(12):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for idx in range(n_clusters):
            member_values = values[labels == idx]
            if member_values.size:
                updated[idx] = float(np.mean(member_values))
        if np.allclose(updated, centers, atol=1e-3):
            break
        centers = updated
    return np.sort(centers)


def sort_corners_to_grid(corners: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """将角点分配到固定的行列网格位置，缺失格点保留 NaN。"""
    rows, cols = pattern_size
    total_expected = rows * cols
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return _empty_grid(pattern_size)

    row_axis, col_axis = _estimate_grid_axes(pts, pattern_size)
    row_proj = pts @ row_axis
    col_proj = pts @ col_axis
    col_centers = _cluster_axis(row_proj, cols)
    row_centers = _cluster_axis(col_proj, rows)

    grid_corners = np.full((total_expected, 1, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros(total_expected, dtype=bool)
    assignment_cost = np.full(total_expected, np.inf, dtype=np.float32)

    col_indices = np.argmin(np.abs(row_proj[:, None] - col_centers[None, :]), axis=1)
    row_indices = np.argmin(np.abs(col_proj[:, None] - row_centers[None, :]), axis=1)

    for pt_idx, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
        grid_idx = int(row_idx) * cols + int(col_idx)
        cost = abs(row_proj[pt_idx] - col_centers[col_idx]) + abs(col_proj[pt_idx] - row_centers[row_idx])
        if cost < assignment_cost[grid_idx]:
            assignment_cost[grid_idx] = cost
            grid_corners[grid_idx, 0] = pts[pt_idx]
            valid_mask[grid_idx] = True

    return grid_corners, valid_mask


def _median_spacing(points: np.ndarray) -> float:
    """估计点集的典型最近邻间距。"""
    pts = points.reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 2:
        return 1.0
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return float(np.median(nearest[np.isfinite(nearest)])) if np.any(np.isfinite(nearest)) else 1.0


def _mad_outliers(values: np.ndarray, thresh: float) -> np.ndarray:
    """基于 MAD 规则识别离群值。"""
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-6:
        absolute_tol = max(1.0, abs(med) * 0.25)
        return np.abs(values - med) > absolute_tol
    return np.abs(values - med) / (1.4826 * mad) > thresh


def _build_quality(
    matched_left: np.ndarray,
    matched_right: np.ndarray,
    expected_count: int,
    initial_count: int,
    outlier_count: int,
) -> Dict[str, float]:
    """构建匹配质量统计信息。"""
    disparities = compute_disparities(matched_left, matched_right)
    y_diff = np.abs(matched_left[:, 0, 1] - matched_right[:, 0, 1]) if len(matched_left) else np.array([])
    return {
        "matched_count": int(len(matched_left)),
        "initial_matched_count": int(initial_count),
        "expected_count": int(expected_count),
        "match_ratio": float(len(matched_left) / expected_count) if expected_count else 0.0,
        "disparity_mean": float(np.mean(disparities)) if disparities.size else 0.0,
        "disparity_std": float(np.std(disparities)) if disparities.size else 0.0,
        "disparity_min": float(np.min(disparities)) if disparities.size else 0.0,
        "disparity_max": float(np.max(disparities)) if disparities.size else 0.0,
        "y_diff_mean": float(np.mean(y_diff)) if y_diff.size else 0.0,
        "y_diff_std": float(np.std(y_diff)) if y_diff.size else 0.0,
        "outlier_count": int(outlier_count),
    }


def _validate_and_filter_matches(
    matched_left: np.ndarray,
    matched_right: np.ndarray,
    expected_count: int,
    config: CornerMatchingConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """对匹配结果执行视差与几何一致性校验。"""
    initial_count = len(matched_left)
    min_required = max(config.min_matched_points, int(np.ceil(expected_count * config.min_match_ratio)))
    if initial_count < min_required:
        raise CornerMatchingError(
            f"匹配点数量不足：匹配到 {initial_count} 对，至少需要 {min_required} 对"
        )

    original_left = matched_left
    original_right = matched_right
    disparities = compute_disparities(matched_left, matched_right)
    outliers = np.zeros(initial_count, dtype=bool)

    if config.require_positive_disparity:
        positive_bad = disparities <= config.min_positive_disparity
        outliers |= positive_bad
        if np.all(positive_bad):
            message = "所有匹配点视差均为非正值，请检查左右图顺序或匹配排序"
            if config.strict_geometry:
                raise CornerMatchingError(f"几何校验失败：{message}")
            logger.warning("%s；当前为兼容模式，继续保留匹配点", message)

    outliers |= _mad_outliers(disparities, config.disparity_mad_thresh)

    y_diff = np.abs(matched_left[:, 0, 1] - matched_right[:, 0, 1])
    spacing = max(_median_spacing(matched_left), 1.0)
    y_thresh = spacing * config.y_diff_spacing_factor
    outliers |= y_diff > y_thresh

    outlier_count = int(np.count_nonzero(outliers))
    outlier_ratio = outlier_count / max(initial_count, 1)
    if outlier_ratio > config.max_outlier_ratio:
        message = f"异常匹配比例过高 {outlier_ratio:.1%}，异常点 {outlier_count}/{initial_count}"
        if config.strict_geometry:
            raise CornerMatchingError(f"几何校验失败：{message}")
        logger.warning("%s；当前为兼容模式，不剔除整批匹配点", message)
        quality = _build_quality(matched_left, matched_right, expected_count, initial_count, outlier_count)
        return matched_left, matched_right, quality

    if outlier_count and config.filter_outliers:
        logger.warning("filtered %d geometric outlier matches", outlier_count)
        keep = ~outliers
        matched_left = matched_left[keep]
        matched_right = matched_right[keep]
        if len(matched_left) < min_required:
            message = f"剔除异常匹配后点数不足：剩余 {len(matched_left)} 对，至少需要 {min_required} 对"
            if config.strict_geometry:
                raise CornerMatchingError(message)
            logger.warning("%s；当前为兼容模式，恢复未剔除匹配点", message)
            matched_left = original_left
            matched_right = original_right

    quality = _build_quality(matched_left, matched_right, expected_count, initial_count, outlier_count)
    return matched_left, matched_right, quality


def match_by_relative_coordinates(
    corners_left: np.ndarray,
    corners_right: np.ndarray,
    pattern_size: Tuple[int, int],
    config: Optional[CornerMatchingConfig] = None,
    return_quality: bool = False,
):
    """根据稳定的网格位置对左右角点进行匹配。"""
    config = config or DEFAULT_CONFIG.corner_matching
    expected_count = pattern_size[0] * pattern_size[1]

    if corners_left is None or len(corners_left) == 0:
        raise CornerMatchingError("左图角点为空，无法匹配")
    if corners_right is None or len(corners_right) == 0:
        raise CornerMatchingError("右图角点为空，无法匹配")

    grid_left, valid_left = sort_corners_to_grid(corners_left, pattern_size)
    grid_right, valid_right = sort_corners_to_grid(corners_right, pattern_size)
    both_valid = valid_left & valid_right

    matched_left = grid_left[both_valid]
    matched_right = grid_right[both_valid]
    matched_left, matched_right, quality = _validate_and_filter_matches(
        matched_left,
        matched_right,
        expected_count,
        config,
    )

    logger.info(
        "corner matching complete: matched=%d/%d ratio=%.2f disparity_mean=%.3f outliers=%d",
        quality["matched_count"],
        expected_count,
        quality["match_ratio"],
        quality["disparity_mean"],
        quality["outlier_count"],
    )

    if return_quality:
        return matched_left, matched_right, quality
    return matched_left, matched_right


def compute_disparities(corners_left: np.ndarray, corners_right: np.ndarray) -> np.ndarray:
    """计算左右角点的水平视差。"""
    x_left = corners_left[:, 0, 0]
    x_right = corners_right[:, 0, 0]
    return x_left - x_right
