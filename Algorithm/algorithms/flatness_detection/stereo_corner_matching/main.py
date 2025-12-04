"""
双目立体视觉角点匹配模块

功能：从左右两张处理后的棋盘格灰度图中检测角点，按顺序匹配对应角点，并计算视差。

输入要求：
1) 棋盘格灰度图（放在 data 文件夹下）：
   - 左图：文件名包含 "left"（如 left.png）
   - 右图：文件名包含 "right"（如 right.png）

2) 掩膜文件（必须，放在 data 文件夹下）：
   - 左图掩膜：left_mask.png 
   - 右图掩膜：right_mask.png 

3) 棋盘格参数（在 main() 函数中设置）：
   - chessboard_size: (行数, 列数)，例如 (7, 10) 表示 8×11 棋盘格（内角点）

输出文件（保存到 result 文件夹）：
- corners_left.json：左图角点坐标（像素）
- corners_right.json：右图角点坐标（像素）
- disparities.json：视差值（像素）
- matches_visualization.png：匹配可视化图像

处理流程：
1) 读取左右棋盘格灰度图和掩膜
2) 使用 OpenCV 检测棋盘格角点（仅在掩膜区域内）
3) 亚像素精化角点位置
4) 按行排序角点（从上到下，从左到右）
5) 按索引一一对应匹配角点
6) 计算视差（disparity = x_left - x_right）
7) 保存结果和可视化图像
"""
import os
import json
import glob
from typing import Tuple, Optional

import cv2
import numpy as np


def read_image_grayscale(path: str) -> np.ndarray:
    """读取灰度图片。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    
    return img


def enhance_chessboard_region(img: np.ndarray) -> np.ndarray:
    """
    增强棋盘格区域的对比度。
    
    使用 CLAHE（对比度受限的自适应直方图均衡化）算法增强图像对比度，
    提高棋盘格角点的可见性，从而提高角点检测的成功率。
    
    参数：
        img: 输入灰度图像
    
    返回：
        增强后的灰度图像，与输入图像形状相同
    
    说明：
        - clipLimit=2.0: 对比度限制参数，防止过度增强
        - tileGridSize=(8, 8): 将图像分成 8×8 的网格，对每个网格独立进行直方图均衡化
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    return enhanced


def find_chessboard_corners(img: np.ndarray, pattern_size: Tuple[int, int], 
                            allow_partial: bool = True, enhance: bool = False,
                            mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    检测棋盘格角点。
    
    使用 OpenCV 的 findChessboardCorners 方法检测棋盘格内角点。
    支持部分棋盘格检测和掩膜限制。
    
    参数：
        img: 输入灰度图像
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
        allow_partial: 是否允许部分棋盘格检测（图像中只显示部分棋盘格时）
        enhance: 是否增强图像对比度（使用 CLAHE），默认 False
        mask: 可选的掩膜图像，用于限制角点检测区域
    
    返回：
        检测到的角点坐标数组，形状是 (N, 1, 2)，如果检测失败则返回 None
    """
    # 图像预处理
    if enhance:
        img_processed = enhance_chessboard_region(img)
    else:
        img_processed = img.copy()
    
    # 应用掩膜：限制角点检测只在掩膜区域内进行
    if mask is not None:
        if mask.shape != img_processed.shape:
            mask = cv2.resize(mask, (img_processed.shape[1], img_processed.shape[0]))
        img_processed = cv2.bitwise_and(img_processed, mask)
    
    # 优先使用 findChessboardCorners 检测角点
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if allow_partial:
        partial_flag = getattr(cv2, "CALIB_CB_PARTIAL_OK", 0)
        if partial_flag:
            flags += partial_flag
    
    ret = False
    corners: Optional[np.ndarray] = None
    
    ret, corners = cv2.findChessboardCorners(
        img_processed,
        pattern_size,
        flags=flags
    )
    
    if ret:
        print(f"  [提示] findChessboardCorners 检测成功，检测到 {len(corners)} 个角点。")
    else:
        # 如果常规方法失败，尝试使用 findChessboardCornersSB（如果可用）
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
                sb_result = cv2.findChessboardCornersSB(
                    img_processed,
                    pattern_size,
                    flags=sb_flags
                )
                if isinstance(sb_result, tuple):
                    sb_ret, sb_corners = sb_result
                else:
                    sb_ret, sb_corners = True, sb_result
                
                if sb_ret and sb_corners is not None:
                    sb_corners = np.asarray(sb_corners, dtype=np.float32)
                    if sb_corners.ndim == 2:
                        sb_corners = sb_corners.reshape(-1, 1, 2)
                    if sb_corners.shape[0] == pattern_size[0] * pattern_size[1]:
                        corners = sb_corners
                        ret = True
                        print(f"  [提示] findChessboardCorners 失败，使用 findChessboardCornersSB 成功检测到 {len(corners)} 个角点。")
            except Exception:
                pass
    
    if not ret:
        expected_count = pattern_size[0] * pattern_size[1]
        print(f"  [错误] 角点检测失败：未检测到任何角点（期望 {expected_count} 个）")
        return None
    
    # 验证角点是否在掩膜区域内
    if mask is not None:
        corners_pts = corners.reshape(-1, 2).astype(np.int32)
        valid_mask = np.zeros(len(corners_pts), dtype=bool)
        for i, pt in enumerate(corners_pts):
            y, x = int(pt[1]), int(pt[0])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    valid_mask[i] = True
        
        valid_count = valid_mask.sum()
        expected_count = pattern_size[0] * pattern_size[1]
        min_required = int(expected_count * 0.3)
        
        if valid_count < min_required:
            print(f"  [错误] 掩膜验证失败：检测到 {len(corners_pts)} 个角点，"
                  f"但只有 {valid_count} 个在掩膜区域内（需要至少 {min_required} 个）")
            return None
        
        corners = corners[valid_mask]
        if len(corners) < expected_count:
            print(f"  [提示] 检测到 {len(corners)} 个有效角点（期望 {expected_count} 个）")
    
    return corners


def refine_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    亚像素精化角点位置。
    
    使用 cornerSubPix 函数将角点位置从整数像素精化到亚像素级（0.1像素精度），
    基于局部灰度梯度优化，提高角点定位精度。
    
    参数：
        img: 灰度图像（numpy 数组）
        corners: 初始角点坐标，形状是 (N, 1, 2)
    
    返回：
        精化后的角点坐标，形状与输入相同。
    """
    # 设置亚像素精化的搜索窗口和终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    winSize = (11, 11)  # 搜索窗口大小
    zeroZone = (-1, -1)  # 死区大小（不使用）
    
    # 执行亚像素精化
    refined = cv2.cornerSubPix(img, corners, winSize, zeroZone, criteria)
    
    return refined


def sort_corners_to_grid(corners: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将角点按行列分配到固定大小的网格中，处理缺失角点的情况。
    
    使用统一的排序方法：
    1. 按 y 坐标排序，将角点分组为行
    2. 对每一行内的角点，按 x 坐标排序，分配到列
    3. 将角点分配到固定大小的网格中，缺失位置用 NaN 填充
    
    参数：
        corners: 角点坐标数组，形状是 (N, 1, 2)
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
    
    返回：
        (grid_corners, valid_mask)
        - grid_corners: 固定大小的角点数组，形状是 (num_rows * num_cols, 1, 2)，
                       缺失的角点位置用 NaN 填充
        - valid_mask: 布尔数组，标记哪些位置有有效的角点
    """
    pts = corners.reshape(-1, 2)  # (N, 2)
    num_rows, num_cols = pattern_size
    total_expected = num_rows * num_cols
    
    if len(pts) == 0:
        # 如果没有角点，返回全 NaN 的数组
        grid_corners = np.full((total_expected, 1, 2), np.nan, dtype=np.float32)
        valid_mask = np.zeros(total_expected, dtype=bool)
        return grid_corners, valid_mask
    
    all_x = pts[:, 0]
    all_y = pts[:, 1]
    
    # 创建固定大小的网格，初始为 NaN
    grid_corners = np.full((total_expected, 1, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros(total_expected, dtype=bool)
    
    # 步骤1：按 y 坐标排序，将角点分组为行
    y_sorted_indices = np.argsort(all_y)
    row_groups = []
    for i in range(num_rows):
        start_idx = int(i * len(y_sorted_indices) / num_rows)
        end_idx = int((i + 1) * len(y_sorted_indices) / num_rows)
        row_groups.append(y_sorted_indices[start_idx:end_idx])
    
    # 步骤2：对每一行，按 x 坐标排序，分配到列
    for row_idx, row_indices in enumerate(row_groups):
        if len(row_indices) == 0:
            continue
        
        row_x = all_x[row_indices]
        x_sorted_indices = [row_indices[i] for i in np.argsort(row_x)]
        
        # 将这一行的角点分配到列
        for col_idx in range(num_cols):
            start_idx = int(col_idx * len(x_sorted_indices) / num_cols)
            end_idx = int((col_idx + 1) * len(x_sorted_indices) / num_cols)
            if start_idx < len(x_sorted_indices):
                pt_idx = x_sorted_indices[start_idx]
                grid_idx = row_idx * num_cols + col_idx
                grid_corners[grid_idx] = corners[pt_idx]
                valid_mask[grid_idx] = True
    
    return grid_corners, valid_mask


def match_by_relative_coordinates(
    corners_left: np.ndarray, 
    corners_right: np.ndarray,
    pattern_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    角点匹配：使用网格排序方法，按行列位置匹配，处理缺失角点的情况。
    
    将左右图的角点都分配到固定大小的网格中，然后按索引匹配。
    只有两个图中都存在的角点位置才会被匹配。
    
    参数：
        corners_left: 左图角点坐标，形状是 (N, 1, 2)
        corners_right: 右图角点坐标，形状是 (M, 1, 2)
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
    
    返回：
        (matched_left, matched_right)
        - matched_left: 匹配成功的左图角点数组（只包含两个图都有的角点）
        - matched_right: 匹配成功的右图角点数组（只包含两个图都有的角点）
    """
    # 将角点分配到固定大小的网格中
    grid_left, valid_left = sort_corners_to_grid(corners_left, pattern_size)
    grid_right, valid_right = sort_corners_to_grid(corners_right, pattern_size)
    
    # 找到两个图都有效的角点位置
    both_valid = valid_left & valid_right
    
    # 提取匹配的角点
    matched_left = grid_left[both_valid]
    matched_right = grid_right[both_valid]
    
    num_matched = np.sum(both_valid)
    num_total = len(both_valid)
    num_missing = num_total - num_matched
    
    print(f"  角点匹配成功: {num_matched} 对（共 {num_total} 个位置，缺失 {num_missing} 个）")
    
    return matched_left, matched_right


def compute_disparities(corners_left: np.ndarray, corners_right: np.ndarray) -> np.ndarray:
    """计算视差值。"""
    x_left = corners_left[:, 0, 0]
    x_right = corners_right[:, 0, 0]
    disparities = x_left - x_right
    return disparities


def visualize_matches(img_left: np.ndarray,
                      img_right: np.ndarray,
                      corners_left: np.ndarray,
                      corners_right: np.ndarray,
                      disparities: np.ndarray,
                      out_path: str) -> None:
    """生成匹配角点与视差的可视化图。"""
    if corners_left.size == 0 or corners_right.size == 0 or disparities.size == 0:
        print("  [提示] 没有可视化的匹配角点，跳过可视化。")
        return

    if img_left.ndim == 2:
        left_vis = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = img_left.copy()

    if img_right.ndim == 2:
        right_vis = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = img_right.copy()

    h = max(left_vis.shape[0], right_vis.shape[0])
    w = left_vis.shape[1] + right_vis.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:left_vis.shape[0], :left_vis.shape[1]] = left_vis
    canvas[:right_vis.shape[0], left_vis.shape[1]:left_vis.shape[1] + right_vis.shape[1]] = right_vis

    offset_x = left_vis.shape[1]
    disp_min = float(np.min(disparities))
    disp_max = float(np.max(disparities))
    disp_span = max(disp_max - disp_min, 1e-6)

    corners_left_flat = corners_left.reshape(-1, 2)
    corners_right_flat = corners_right.reshape(-1, 2)

    for idx, (pt_l, pt_r, disp) in enumerate(zip(corners_left_flat, corners_right_flat, disparities)):
        alpha = (float(disp) - disp_min) / disp_span
        color = (
            int(255 * (1.0 - alpha)),
            int(128),
            int(255 * alpha)
        )

        pt_l_int = tuple(np.round(pt_l).astype(int))
        pt_r_int = tuple(np.round(pt_r).astype(int))
        pt_r_shifted = (pt_r_int[0] + offset_x, pt_r_int[1])

        cv2.circle(canvas, pt_l_int, 5, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_r_shifted, 5, color, 2, lineType=cv2.LINE_AA)

        # 为所有点添加标号
        pair_id = str(idx + 1)
        cv2.putText(canvas, pair_id, (pt_l_int[0] + 6, pt_l_int[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, pair_id, (pt_r_shifted[0] + 6, pt_r_shifted[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def find_input_images(data_dir: str) -> Tuple[str, str]:
    """在 data 目录里查找文件名包含 "left" 和 "right" 的棋盘格图片。"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, e)))
    
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到图片")
    
    left_candidates = sorted([f for f in files if "left" in os.path.basename(f).lower()])
    right_candidates = sorted([f for f in files if "right" in os.path.basename(f).lower()])
    
    if not left_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'left' 的图片")
    if not right_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'right' 的图片")
    
    return left_candidates[0], right_candidates[0]


def find_mask_images(data_dir: str, left_path: str, right_path: str) -> Tuple[Optional[str], Optional[str]]:
    """在 data 目录中查找掩膜文件。"""
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    
    def _find_mask(img_path: str, default_prefix: str) -> Optional[str]:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = []
        for ext in exts:
            candidates.append(os.path.join(data_dir, f"{default_prefix}{ext}"))
            candidates.append(os.path.join(data_dir, f"{base_name}_mask{ext}"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None
    
    left_mask = _find_mask(left_path, "left_mask")
    right_mask = _find_mask(right_path, "right_mask")
    
    return left_mask, right_mask


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
                       min_ratio: float = 0.12,
                       padding: int = 50,
                       chessboard_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    """
    根据掩膜精确裁剪图像到棋盘格区域。
    
    使用结构形态 + 几何特征 + 内部模式验证的方法（detect_chessboard_mask）识别棋盘格区域。
    
    参数：
        img: 输入图像
        mask: 掩膜图像（棋盘格掩膜，白色区域表示棋盘格位置，内部有黑色矩形）
        min_ratio: 掩膜覆盖比例阈值（已弃用，总是裁剪）
        padding: 裁剪边界的额外边距（像素）
        chessboard_size: 棋盘格内角点数量 (行数, 列数)，已弃用
    
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


def save_results(out_dir: str, corners_left: np.ndarray, corners_right: np.ndarray, 
                 disparities: np.ndarray) -> None:
    """保存角点坐标和视差值到 JSON 文件。"""
    left_pts = corners_left.reshape(-1, 2).tolist()
    right_pts = corners_right.reshape(-1, 2).tolist()
    disparities_list = disparities.tolist()
    
    left_path = os.path.join(out_dir, "corners_left.json")
    with open(left_path, 'w', encoding='utf-8') as f:
        json.dump(left_pts, f, indent=2, ensure_ascii=False)
    
    right_path = os.path.join(out_dir, "corners_right.json")
    with open(right_path, 'w', encoding='utf-8') as f:
        json.dump(right_pts, f, indent=2, ensure_ascii=False)
    
    disp_path = os.path.join(out_dir, "disparities.json")
    with open(disp_path, 'w', encoding='utf-8') as f:
        json.dump(disparities_list, f, indent=2, ensure_ascii=False)
    
    print(f"保存结果:\n  左图角点: {left_path}\n  右图角点: {right_path}\n  视差值: {disp_path}")


def main():
    """主函数：执行完整的角点检测、匹配和视差计算流程。"""
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    left_path, right_path = find_input_images(data_dir)
    img_left = read_image_grayscale(left_path)
    img_right = read_image_grayscale(right_path)
    
    print(f"读取图片:\n  左图: {left_path}\n  右图: {right_path}")
    print(f"图像尺寸: 左图 {img_left.shape}, 右图 {img_right.shape}")

    chessboard_size = (7, 10)
    print(f"棋盘格参数: {chessboard_size[0]}×{chessboard_size[1]} (内角点数)")

    print("\n查找棋盘格掩膜...")
    left_mask_path, right_mask_path = find_mask_images(data_dir, left_path, right_path)
    
    if left_mask_path and os.path.exists(left_mask_path):
        print(f"  读取左图掩膜: {left_mask_path}")
        mask_left = read_image_grayscale(left_mask_path)
    else:
        raise FileNotFoundError("未找到左图掩膜文件！")
    
    if right_mask_path and os.path.exists(right_mask_path):
        print(f"  读取右图掩膜: {right_mask_path}")
        mask_right = read_image_grayscale(right_mask_path)
    else:
        raise FileNotFoundError("未找到右图掩膜文件！")
    
    mask_area_left = np.sum(mask_left > 0) / (mask_left.shape[0] * mask_left.shape[1]) * 100
    mask_area_right = np.sum(mask_right > 0) / (mask_right.shape[0] * mask_right.shape[1]) * 100
    print(f"  左图掩膜覆盖: {mask_area_left:.1f}%")
    print(f"  右图掩膜覆盖: {mask_area_right:.1f}%")

    print("\n裁剪左图...")
    img_left_det, mask_left_det, left_offset, left_ratio = crop_image_by_mask(
        img_left, mask_left, min_ratio=0.12, padding=50, chessboard_size=chessboard_size)
    
    # 保存裁剪后的左图掩码
    cropped_left_mask_path = os.path.join(out_dir, "cropped_left_mask.png")
    cv2.imwrite(cropped_left_mask_path, mask_left_det)
    print(f"  保存左图裁剪掩码: {cropped_left_mask_path}")

    print("\n裁剪右图...")
    img_right_det, mask_right_det, right_offset, right_ratio = crop_image_by_mask(
        img_right, mask_right, min_ratio=0.12, padding=50, chessboard_size=chessboard_size)
    
    # 保存裁剪后的右图掩码
    cropped_right_mask_path = os.path.join(out_dir, "cropped_right_mask.png")
    cv2.imwrite(cropped_right_mask_path, mask_right_det)
    print(f"  保存右图裁剪掩码: {cropped_right_mask_path}")

    allow_partial = True
    enhance_contrast = False  # 统一不使用对比度增强
    
    print("\n检测左图角点...")
    corners_left = find_chessboard_corners(img_left_det, chessboard_size, 
                                          allow_partial=allow_partial, 
                                          enhance=enhance_contrast,
                                          mask=mask_left_det)
    if corners_left is None:
        raise RuntimeError("左图角点检测失败！")
    if left_offset != (0, 0):
        corners_left[:, 0, 0] += left_offset[0]
        corners_left[:, 0, 1] += left_offset[1]
    
    print("检测右图角点...")
    corners_right = find_chessboard_corners(img_right_det, chessboard_size,
                                            allow_partial=allow_partial,
                                            enhance=enhance_contrast,
                                            mask=mask_right_det)
    if corners_right is None:
        raise RuntimeError("右图角点检测失败！")
    if right_offset != (0, 0):
        corners_right[:, 0, 0] += right_offset[0]
        corners_right[:, 0, 1] += right_offset[1]
    
    print(f"检测到角点: 左图 {len(corners_left)} 个, 右图 {len(corners_right)} 个")

    print("\n亚像素精化角点位置...")
    corners_left = refine_corners(img_left, corners_left)
    corners_right = refine_corners(img_right, corners_right)

    # ========== 角点匹配 ==========
    print("\n进行角点匹配...")
    corners_left_matched, corners_right_matched = match_by_relative_coordinates(
        corners_left, corners_right, chessboard_size
    )
    
    num_matched = len(corners_left_matched)
    print(f"匹配成功: {num_matched} 对")
    
    if num_matched == 0:
        raise RuntimeError("角点匹配未找到任何匹配对！")

    # ========== 计算视差 ==========
    print("\n计算视差...")
    disparities = compute_disparities(corners_left_matched, corners_right_matched)
    
    print(f"视差统计:")
    print(f"  最小值: {disparities.min():.2f} 像素")
    print(f"  最大值: {disparities.max():.2f} 像素")
    print(f"  平均值: {disparities.mean():.2f} 像素")
    print(f"  标准差: {disparities.std():.2f} 像素")

    # ========== 保存结果 ==========
    print("\n保存结果...")
    save_results(out_dir, corners_left_matched, corners_right_matched, disparities)

    # ========== 可视化匹配 ==========
    print("\n生成匹配可视化图...")
    match_vis_path = os.path.join(out_dir, "matches_visualization.png")
    visualize_matches(
        img_left,
        img_right,
        corners_left_matched,
        corners_right_matched,
        disparities,
        match_vis_path
    )
    print(f"  匹配可视化: {match_vis_path}")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

