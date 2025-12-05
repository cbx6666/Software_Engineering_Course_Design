"""
图像处理相关函数
"""
from typing import Tuple, Optional
import cv2
import numpy as np


def detect_chessboard_mask(mask: np.ndarray, min_area: int = 5000) -> Optional[Tuple[int, int, int, int]]:
    """
    从混杂的掩码图中准确识别棋盘格区域。
    
    使用结构形态 + 几何特征 + 内部模式验证的方法，从一堆"其他白色区域 + 黑白噪声"掩码里，
    准确找出棋盘格掩码。棋盘格掩码的特征：白色大矩形 + 规则黑白方格（白底 + 黑色正方形）。
    
    过滤策略：
    1. 面积过滤：面积 > 阈值（排除小噪声）
    2. 形状过滤：轮廓近似 4 边形（棋盘格是矩形）
    3. 宽高比过滤：宽高比合理（0.6-2.0）
    4. 内部模式验证：行/列黑色分布呈周期性（最强特征，几乎无法伪造）
    
    参数：
        mask: 掩码图像（二值图，255=白色区域，0=黑色区域）
        min_area: 最小面积阈值，默认 5000 像素
    
    返回：
        如果检测成功，返回 (x, y, w, h)，否则返回 None
    """
    if mask is None:
        return None
    
    # 确保是二值图
    if len(mask.shape) == 3:
        mask_binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_binary = mask.copy()
    
    # 二值化：确保是 0 和 255
    _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)
    
    # 查找所有轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for cnt in contours:
        # 过滤 1：面积 > 阈值
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 过滤 2：形状接近四边形
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        
        # 过滤 3：宽高比合理
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h if h > 0 else 0
        if not (0.6 < ratio < 2.0):
            continue
        
        # 过滤 4：内部周期验证（最强特征）
        # 裁剪出区域
        crop = mask_binary[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        
        # 统计每列和每行的黑色数量（0值）
        col_sum = np.sum(crop == 0, axis=0)  # 每列的黑色像素数
        row_sum = np.sum(crop == 0, axis=1)  # 每行的黑色像素数
        
        # 检测周期性峰值
        # 棋盘格的列方向应该有多个峰值（对应多列黑色方格）
        col_distance = max(w // 10, 5)  # 峰值之间的最小距离
        row_distance = max(h // 10, 5)
        
        # 使用简单的峰值检测方法
        col_peaks = []
        row_peaks = []
        for i in range(1, len(col_sum) - 1):
            if col_sum[i] > col_sum[i-1] and col_sum[i] > col_sum[i+1] and col_sum[i] > np.mean(col_sum):
                col_peaks.append(i)
        for i in range(1, len(row_sum) - 1):
            if row_sum[i] > row_sum[i-1] and row_sum[i] > row_sum[i+1] and row_sum[i] > np.mean(row_sum):
                row_peaks.append(i)
        
        # 简单的距离过滤
        if col_distance and len(col_peaks) > 1:
            filtered = [col_peaks[0]]
            for p in col_peaks[1:]:
                if p - filtered[-1] >= col_distance:
                    filtered.append(p)
            col_peaks = filtered
        if row_distance and len(row_peaks) > 1:
            filtered = [row_peaks[0]]
            for p in row_peaks[1:]:
                if p - filtered[-1] >= row_distance:
                    filtered.append(p)
            row_peaks = filtered
        
        col_peaks = np.array(col_peaks)
        row_peaks = np.array(row_peaks)
        
        # 棋盘格应该有至少 3-4 个峰值（对应多行/多列）
        if len(col_peaks) >= 3 and len(row_peaks) >= 3:
            candidates.append((cnt, x, y, w, h, area))
    
    # 返回面积最大的候选区域
    if candidates:
        # 按面积排序，选择最大的
        candidates = sorted(candidates, key=lambda c: c[5], reverse=True)
        _, x, y, w, h, _ = candidates[0]
        return (x, y, w, h)
    
    return None


def crop_image_by_mask(img: np.ndarray,
                       mask: np.ndarray,
                       padding: int = 50) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    """
    根据掩膜精确裁剪图像到棋盘格区域。
    
    使用结构形态 + 几何特征 + 内部模式验证的方法（detect_chessboard_mask）识别棋盘格区域。
    
    参数：
        img: 输入图像
        mask: 掩膜图像（棋盘格掩膜，白色区域表示棋盘格位置，内部有黑色矩形）
        padding: 裁剪边界的额外边距（像素）
    
    返回：
        (裁剪后的图像, 裁剪后的掩膜, (x偏移, y偏移), 掩膜覆盖比例)
    """
    if mask is None or img is None:
        return img, mask, (0, 0), 0.0

    mask_binary = (mask > 0).astype(np.uint8)
    ratio = float(np.sum(mask_binary)) / mask_binary.size
    if ratio == 0:
        return img, mask, (0, 0), 0.0

    mask_filtered = (mask_binary * 255).astype(np.uint8)
    
    # 使用结构形态 + 几何特征 + 内部模式验证的方法
    bbox = detect_chessboard_mask(mask)
    if bbox is None:
        print(f"  [警告] 无法检测到棋盘格区域，返回原图")
        return img, mask_filtered, (0, 0), ratio
    
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    detection_method = "结构特征检测"

    # 添加 padding 并确保不越界
    y1_orig, y2_orig, x1_orig, x2_orig = y1, y2, x1, x2
    y1 = max(y1 - padding, 0)
    y2 = min(y2 + padding, img.shape[0])
    x1 = max(x1 - padding, 0)
    x2 = min(x2 + padding, img.shape[1])

    # 执行裁剪
    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask_filtered[y1:y2, x1:x2]
    
    # 计算裁剪后掩膜覆盖率（用于信息输出）
    cropped_mask_binary = (cropped_mask > 0).astype(np.uint8)
    cropped_ratio = float(np.sum(cropped_mask_binary)) / cropped_mask_binary.size * 100
    
    # 统一输出最终裁剪信息
    print(f"  [提示] 通过{detection_method}定位到区域: x[{x1_orig}:{x2_orig}], y[{y1_orig}:{y2_orig}]")
    print(f"  [提示] 添加padding后裁剪到: x[{x1}:{x2}], y[{y1}:{y2}], 掩膜覆盖率: {cropped_ratio:.1f}%")

    return cropped_img, cropped_mask, (x1, y1), cropped_ratio / 100.0

