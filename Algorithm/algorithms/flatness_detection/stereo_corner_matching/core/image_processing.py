"""角点检测前的图像区域定位工具。"""

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ...config import DEFAULT_CONFIG, CornerDetectionConfig
    from ...logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, CornerDetectionConfig
    from logging_utils import get_logger


logger = get_logger("corners.image")


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    """将输入掩膜统一转换为二值图。"""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask.astype(np.uint8, copy=False), 127, 255, cv2.THRESH_BINARY)
    return mask


def _nonzero_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """返回掩膜非零区域的最小包围框。"""
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def detect_chessboard_mask(
    mask: np.ndarray,
    min_area: int = None,
    config: CornerDetectionConfig = None,
) -> Optional[Tuple[int, int, int, int]]:
    """从实心掩膜中定位棋盘格候选框。"""
    config = config or DEFAULT_CONFIG.corner_detection
    min_area = config.crop_min_area if min_area is None else min_area
    mask_binary = _binary_mask(mask)
    if mask_binary is None:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    candidates = []
    min_ratio, max_ratio = config.crop_aspect_ratio_range

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area or w <= 0 or h <= 0:
            continue

        aspect = w / float(h)
        if aspect < min_ratio or aspect > max_ratio:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < config.crop_component_min_fill_ratio:
            continue

        score = float(area) * float(fill_ratio)
        candidates.append((score, x, y, w, h))

    if candidates:
        candidates.sort(reverse=True)
        _, x, y, w, h = candidates[0]
        return int(x), int(y), int(w), int(h)

    return _nonzero_bbox(mask_binary)


def crop_image_by_mask(
    img: np.ndarray,
    mask: np.ndarray,
    padding: int = None,
    config: CornerDetectionConfig = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    """按掩膜定位结果裁剪图像，并返回偏移量和覆盖率。"""
    config = config or DEFAULT_CONFIG.corner_detection
    padding = config.crop_padding if padding is None else padding

    if mask is None or img is None:
        return img, mask, (0, 0), 0.0

    mask_binary = (_binary_mask(mask) > 0).astype(np.uint8)
    ratio = float(np.count_nonzero(mask_binary)) / mask_binary.size
    if ratio == 0:
        return img, mask, (0, 0), 0.0

    mask_filtered = (mask_binary * 255).astype(np.uint8)
    bbox = detect_chessboard_mask(mask_filtered, config=config)
    if bbox is None:
        logger.warning("无法定位到棋盘格区域，返回原图")
        return img, mask_filtered, (0, 0), ratio

    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    x1_orig, x2_orig, y1_orig, y2_orig = x1, x2, y1, y2
    y1 = max(y1 - padding, 0)
    y2 = min(y2 + padding, img.shape[0])
    x1 = max(x1 - padding, 0)
    x2 = min(x2 + padding, img.shape[1])

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask_filtered[y1:y2, x1:x2]

    cropped_mask_binary = (cropped_mask > 0).astype(np.uint8)
    cropped_ratio = float(np.count_nonzero(cropped_mask_binary)) / cropped_mask_binary.size * 100

    logger.info("通过连通域定位到区域: x[%d:%d], y[%d:%d]", x1_orig, x2_orig, y1_orig, y2_orig)
    logger.info("添加 padding 后裁剪到: x[%d:%d], y[%d:%d], 掩膜覆盖率: %.1f%%", x1, x2, y1, y2, cropped_ratio)
    return cropped_img, cropped_mask, (x1, y1), cropped_ratio / 100.0
