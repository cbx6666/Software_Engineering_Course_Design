"""
角点检测相关函数
"""
from typing import Tuple, Optional
import cv2
import numpy as np


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
    corners = None

    # 多尺度尝试，提高 SB 成功率
    scales = [1.0, 1.25, 1.5, 0.8]

    # 多 flag 尝试（SB 对 flags 极度敏感）
    sb_flag_list = [
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE,
    ]

    # 尝试所有组合
    for scale in scales:
        if scale == 1.0:
            img_scaled = img_processed
        else:
            img_scaled = cv2.resize(
                img_processed,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR
            )

        for sb_flags in sb_flag_list:

            try:
                sb_result = cv2.findChessboardCornersSB(
                    img_scaled,
                    pattern_size,
                    flags=sb_flags
                )
            except Exception:
                continue

            # SB 返回值格式兼容
            if isinstance(sb_result, tuple):
                sb_ret, sb_corr = sb_result
            else:
                sb_ret, sb_corr = True, sb_result

            if sb_ret and sb_corr is not None:
                sb_corr = np.asarray(sb_corr, dtype=np.float32)
                if sb_corr.ndim == 2:
                    sb_corr = sb_corr.reshape(-1, 1, 2)

                # 检查数量是否匹配
                detected_count = sb_corr.shape[0]
                expected_count = pattern_size[0] * pattern_size[1]
                
                # 允许部分检测：如果 allow_partial=True，接受至少 50% 的角点
                min_required = expected_count if not allow_partial else max(3, int(expected_count * 0.5))
                
                if detected_count >= min_required:
                    # 恢复坐标
                    if scale != 1.0:
                        sb_corr /= scale

                    corners = sb_corr
                    ret = True
                    print(
                        f"  [提示] SB 检测成功（scale={scale}, flags={sb_flags}），检测到 {detected_count} 个角点（期望 {expected_count} 个）。"
                    )
                    break  # 终止 flags

        if ret:
            break  # 终止 scale

    if not ret:
        expected_count = pattern_size[0] * pattern_size[1]
        print(f"  [错误] SB 角点检测失败（期望 {expected_count} 个）")
        return None
    
    # 返回检测到的角点
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

