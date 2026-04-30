"""棋盘格掩膜处理的基础工具函数。"""

import cv2
import numpy as np


def rect_kernel(size: int) -> np.ndarray:
    """生成奇数边长的矩形结构元素。"""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充二值掩膜内部的孔洞。"""
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_largest_components(mask: np.ndarray, min_area: int, keep_regions: int) -> np.ndarray:
    """保留面积足够且排名靠前的连通域。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[:, cv2.CC_STAT_AREA]
    candidate_labels = [idx for idx in range(1, num_labels) if areas[idx] >= min_area]
    candidate_labels = sorted(candidate_labels, key=lambda idx: areas[idx], reverse=True)
    if keep_regions > 0:
        candidate_labels = candidate_labels[:keep_regions]

    if not candidate_labels:
        return np.zeros_like(mask)

    keep = np.zeros(num_labels, dtype=bool)
    keep[candidate_labels] = True
    return keep[labels].astype(np.uint8) * 255
