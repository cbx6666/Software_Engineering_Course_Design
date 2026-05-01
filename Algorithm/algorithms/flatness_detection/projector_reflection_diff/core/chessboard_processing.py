"""棋盘格区域提取与增强处理。"""

from typing import Tuple

import cv2
import numpy as np

try:
    from ...config import DEFAULT_CONFIG, ProjectionDiffConfig
    from ...logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, ProjectionDiffConfig
    from logging_utils import get_logger

from .chessboard_candidate import finalize_candidate, select_best_candidate
from .chessboard_mask_utils import fill_holes, keep_largest_components, rect_kernel
from .chessboard_texture import build_local_texture_mask, compute_texture_map


logger = get_logger("projection.chess_mask")


def build_chess_mask_from_proj(
    r_proj_gray: np.ndarray,
    block_size: int = None,
    C: int = None,
    min_area: int = None,
    config: ProjectionDiffConfig = None,
) -> np.ndarray:
    """根据投影差分灰度图构建棋盘格掩膜。"""
    config = config or DEFAULT_CONFIG.projection
    block_size = config.adaptive_block_size if block_size is None else block_size
    C = config.adaptive_c if C is None else C
    min_area = config.chess_min_area if min_area is None else min_area

    blur = cv2.GaussianBlur(r_proj_gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        max(3, block_size | 1),
        C,
    )
    if config.chess_open_kernel > 1:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, rect_kernel(config.chess_open_kernel), iterations=1)
    if config.chess_close_kernel > 1:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect_kernel(config.chess_close_kernel), iterations=2)

    adaptive_mask = fill_holes(bw)
    adaptive_mask = keep_largest_components(
        adaptive_mask,
        min_area=min_area,
        keep_regions=config.chess_keep_regions,
    )

    texture_map = compute_texture_map(r_proj_gray, config)
    texture_mask = build_local_texture_mask(texture_map, config)

    adaptive_ratio = float(np.count_nonzero(adaptive_mask)) / adaptive_mask.size
    texture_ratio = float(np.count_nonzero(texture_mask)) / texture_mask.size

    if texture_ratio >= config.min_chess_mask_ratio:
        mask = select_best_candidate(
            r_proj_gray,
            texture_map,
            texture_mask,
            adaptive_mask,
            config,
            logger,
        )
    elif adaptive_ratio >= config.min_chess_mask_ratio:
        logger.warning(
            "纹理掩膜过小（%.4f），回退到自适应掩膜（%.4f）",
            texture_ratio,
            adaptive_ratio,
        )
        mask = finalize_candidate(adaptive_mask, config)
    else:
        logger.warning(
            "纹理掩膜和自适应掩膜都过小（texture=%.4f, adaptive=%.4f），使用纹理掩膜兜底",
            texture_ratio,
            adaptive_ratio,
        )
        mask = finalize_candidate(texture_mask, config)

    mask_ratio = float(np.count_nonzero(mask)) / mask.size
    if mask_ratio < config.min_chess_mask_ratio:
        logger.warning("棋盘格掩膜过小：%.4f", mask_ratio)

    return mask


def suppress_background_in_mask(r_proj_gray: np.ndarray, mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    """仅在棋盘格区域内抑制缓慢变化的背景亮度。"""
    ksize = max(3, ksize | 1)
    low = cv2.GaussianBlur(r_proj_gray, (ksize, ksize), 0)
    filtered = cv2.subtract(r_proj_gray, low)
    out = r_proj_gray.copy().astype(np.int16)
    out[mask > 0] = filtered[mask > 0].astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_chessboard_contrast(
    img: np.ndarray,
    mask: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """仅在棋盘格区域内增强对比度。"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)
    out = img.copy()
    out[mask > 0] = enhanced[mask > 0]
    return out
