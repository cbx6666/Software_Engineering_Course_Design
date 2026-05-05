"""棋盘格候选区域筛选与轻量几何验证。"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from ...config import ProjectionDiffConfig
except ImportError:
    from config import ProjectionDiffConfig

from .chessboard_mask_utils import fill_holes, keep_largest_components, rect_kernel


def nonzero_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """返回非零区域的最小包围框。"""
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def masked_mean_profile(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    """按掩膜权重统计行或列方向的平均响应。"""
    weights = weights.astype(np.float32, copy=False)
    sums = np.sum(values * weights, axis=axis)
    counts = np.sum(weights, axis=axis)
    profile = np.zeros_like(sums, dtype=np.float32)
    valid = counts > 0
    if np.any(valid):
        profile[valid] = sums[valid] / counts[valid]
    return profile


def periodicity_strength(profile: np.ndarray) -> float:
    """估计一维响应是否存在稳定周期。"""
    profile = np.asarray(profile, dtype=np.float32).ravel()
    if profile.size < 8:
        return 0.0

    profile = profile - float(np.mean(profile))
    if np.std(profile) < 1e-6:
        return 0.0

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= np.sum(kernel)
    smooth = np.convolve(profile, kernel, mode="same")
    spectrum = np.abs(np.fft.rfft(smooth))
    if spectrum.size <= 2:
        return 0.0

    band = spectrum[1:max(3, min(spectrum.size, smooth.size // 2 + 1))]
    if band.size == 0:
        return 0.0
    return float(np.max(band) / (np.mean(band) + 1e-6))


def candidate_geometry_metrics(gray: np.ndarray, candidate_mask: np.ndarray) -> Dict[str, float]:
    """计算候选区的梯度平衡性、周期性和轴向对比度。"""
    bbox = nonzero_bbox(candidate_mask)
    if bbox is None:
        return {"gradient_balance": 0.0, "periodicity": 0.0, "axis_contrast": 0.0}

    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w].astype(np.float32, copy=False)
    mask_roi = (candidate_mask[y:y + h, x:x + w] > 0).astype(np.float32)
    if roi.size == 0 or np.count_nonzero(mask_roi) == 0:
        return {"gradient_balance": 0.0, "periodicity": 0.0, "axis_contrast": 0.0}

    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    gx_abs = np.abs(gx)[mask_roi > 0]
    gy_abs = np.abs(gy)[mask_roi > 0]
    mean_gx = float(np.mean(gx_abs)) if gx_abs.size else 0.0
    mean_gy = float(np.mean(gy_abs)) if gy_abs.size else 0.0
    gradient_balance = min(mean_gx, mean_gy) / (max(mean_gx, mean_gy) + 1e-6)

    col_profile = masked_mean_profile(roi, mask_roi, axis=0)
    row_profile = masked_mean_profile(roi, mask_roi, axis=1)
    periodicity = min(periodicity_strength(col_profile), periodicity_strength(row_profile))
    axis_contrast = min(float(np.std(col_profile)), float(np.std(row_profile)))
    return {
        "gradient_balance": gradient_balance,
        "periodicity": periodicity,
        "axis_contrast": axis_contrast,
    }


def finalize_candidate(mask: np.ndarray, config: ProjectionDiffConfig) -> np.ndarray:
    """对候选掩膜做收口、填洞和最终连通域筛选。"""
    out = mask.astype(np.uint8, copy=False)
    if config.chess_close_kernel > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, rect_kernel(config.chess_close_kernel), iterations=1)
    out = fill_holes(out)
    if config.chess_dilate_kernel > 1:
        out = cv2.dilate(out, rect_kernel(config.chess_dilate_kernel), iterations=1)
    return keep_largest_components(out, min_area=config.chess_min_area, keep_regions=1)


def build_candidate_from_component(
    component_mask: np.ndarray,
    adaptive_mask: np.ndarray,
    config: ProjectionDiffConfig,
) -> np.ndarray:
    """根据单个纹理连通域扩展出更完整的候选区域。"""
    expanded = component_mask.copy()
    if config.chess_candidate_expand_kernel > 1:
        expanded = cv2.dilate(expanded, rect_kernel(config.chess_candidate_expand_kernel), iterations=1)

    adaptive_local = cv2.bitwise_and(adaptive_mask, expanded)
    min_overlap = max(config.chess_min_area, int(np.count_nonzero(component_mask) * 0.25))
    if np.count_nonzero(adaptive_local) >= min_overlap:
        candidate = cv2.bitwise_or(component_mask, adaptive_local)
    else:
        candidate = expanded
    return finalize_candidate(candidate, config)


def select_best_candidate(
    gray: np.ndarray,
    texture_map: np.ndarray,
    texture_mask: np.ndarray,
    adaptive_mask: np.ndarray,
    config: ProjectionDiffConfig,
    logger,
) -> np.ndarray:
    """从多个纹理候选中选出最像棋盘格的一块区域。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(texture_mask, connectivity=8)
    candidates = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < config.chess_min_area:
            continue

        component_mask = (labels == idx).astype(np.uint8) * 255
        candidate_mask = build_candidate_from_component(component_mask, adaptive_mask, config)
        candidate_area = int(np.count_nonzero(candidate_mask))
        if candidate_area < config.chess_min_area:
            continue

        ratio = candidate_area / float(candidate_mask.size)
        if ratio > config.chess_mask_max_ratio:
            continue

        fill_ratio = area / float(max(1, w * h))
        texture_score = float(np.mean(texture_map[component_mask > 0]))
        geometry = candidate_geometry_metrics(gray, candidate_mask)
        geometry_valid = (
            texture_score >= config.texture_min_score
            and geometry["gradient_balance"] >= config.geometry_min_gradient_balance
            and geometry["periodicity"] >= config.geometry_min_periodicity
            and geometry["axis_contrast"] >= config.geometry_min_axis_contrast
        )
        score = (
            texture_score
            * max(fill_ratio, 0.1)
            * max(geometry["periodicity"], 0.2)
            * (0.5 + geometry["gradient_balance"])
            * np.sqrt(float(candidate_area))
        )
        candidates.append(
            {
                "mask": candidate_mask,
                "score": float(score),
                "texture_score": texture_score,
                "candidate_area": candidate_area,
                "ratio": ratio,
                "geometry_valid": geometry_valid,
                **geometry,
            }
        )

    if not candidates:
        fallback = adaptive_mask if np.count_nonzero(adaptive_mask) > 0 else texture_mask
        return finalize_candidate(fallback, config)

    valid_candidates = [item for item in candidates if item["geometry_valid"]]
    best_pool = valid_candidates or candidates
    best = max(best_pool, key=lambda item: item["score"])
    logger.info(
        "选中棋盘格候选区: area=%d ratio=%.4f texture=%.1f periodicity=%.2f balance=%.2f valid=%s",
        best["candidate_area"],
        best["ratio"],
        best["texture_score"],
        best["periodicity"],
        best["gradient_balance"],
        best["geometry_valid"],
    )
    return best["mask"]
