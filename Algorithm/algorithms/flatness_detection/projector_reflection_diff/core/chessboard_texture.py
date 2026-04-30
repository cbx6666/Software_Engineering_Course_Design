"""棋盘格纹理响应与初始候选区域提取。"""

import cv2
import numpy as np

try:
    from ...config import ProjectionDiffConfig
except ImportError:
    from config import ProjectionDiffConfig

from .chessboard_mask_utils import keep_largest_components, rect_kernel


def compute_texture_map(r_proj_gray: np.ndarray, config: ProjectionDiffConfig) -> np.ndarray:
    """计算适合小棋盘格场景的纹理热力图。"""
    blur_k = max(3, config.texture_blur_kernel | 1)
    base = cv2.GaussianBlur(r_proj_gray, (blur_k, blur_k), 0)

    window = max(3, config.texture_window_size | 1)
    base_f = base.astype(np.float32, copy=False)
    mean = cv2.boxFilter(base_f, ddepth=-1, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(base_f * base_f, ddepth=-1, ksize=(window, window), normalize=True)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std_map = np.sqrt(variance)
    std_norm = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX)

    lap = np.abs(cv2.Laplacian(base_f, cv2.CV_32F, ksize=3))
    lap_window = max(3, (window // 2) | 1)
    lap_density = cv2.boxFilter(lap, ddepth=-1, ksize=(lap_window, lap_window), normalize=True)
    lap_norm = cv2.normalize(lap_density, None, 0, 255, cv2.NORM_MINMAX)

    texture = cv2.addWeighted(std_norm, 0.45, lap_norm, 0.55, 0.0)
    return np.clip(texture, 0, 255).astype(np.uint8)


def build_local_texture_mask(texture_map: np.ndarray, config: ProjectionDiffConfig) -> np.ndarray:
    """根据纹理热力图提取高响应候选区域。"""
    threshold = float(np.percentile(texture_map, config.texture_percentile))
    mask = (texture_map >= threshold).astype(np.uint8) * 255

    if config.texture_open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect_kernel(config.texture_open_kernel), iterations=1)
    if config.texture_close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect_kernel(config.texture_close_kernel), iterations=2)

    return keep_largest_components(mask, min_area=config.chess_min_area, keep_regions=config.chess_keep_regions)
