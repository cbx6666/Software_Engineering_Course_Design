"""
图像对齐相关函数
"""
from typing import Tuple
import cv2
import numpy as np


def resize_to_match(src: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    调整图片尺寸，让它和另一张图片一样大。
    
    如果两张图片的宽度或高度不一样，先把它们调整成一样大小，才能做后续的对齐和相减操作。
    关灯图和开灯图可能用不同分辨率拍摄，或者相机设置不同，在做像素级操作（比如相减）之前，必须先把尺寸统一。

    参数：
        src: 源图片（numpy 数组）
        target_shape: 目标尺寸，格式是 (高度, 宽度)
    
    返回：
        调整后的图片。如果尺寸已经一样，就直接返回原图（不浪费计算）。
    """
    h, w = target_shape
    
    # 如果尺寸已匹配，直接返回原图
    if src.shape[0] == h and src.shape[1] == w:
        return src
    
    # 使用线性插值调整尺寸
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def ecc_register(moving: np.ndarray, fixed: np.ndarray, motion_model: str = "homography",
                 fast_mode: bool = True, max_iterations: int = 50, eps: float = 1e-4) -> np.ndarray:
    """
    把关灯图的位置"对齐"到开灯图，消除相机轻微移动造成的错位。
    
    即使拍摄时相机有轻微移动或角度变化，也能让两张图片的像素点"对得上"，这样后续相减时不会出现重影或模糊。
    无人机或手持拍摄时，很难保证两张图片完全对齐。如果直接相减，错位的地方会出现"重影"，影响结果质量。
    对齐后，同一位置的像素才真正对应同一块玻璃区域。
    
    优化策略（fast_mode=True）：
        1. 多级优化：先在下采样图像（~800像素）上快速找到初始对齐（约3-5倍加速）
        2. 精细优化：使用初始结果在原图上进行少量迭代优化（提高精度）
        3. 降低迭代次数：从200次降至50次（快速模式）或保持200次（标准模式）
        4. 放宽精度阈值：从1e-6放宽至1e-4（快速模式）或保持1e-6（标准模式）
    
    原理：
        1. （快速模式）对图像进行下采样，在小尺寸上快速计算变换矩阵
        2. 将变换矩阵正确缩放到原图尺寸
        3. （可选）在原图上进行精细优化
        4. 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
    
    参数：
        moving: 需要被对齐的图片（关灯图）
        fixed:  参考图片（开灯图），moving 会被对齐到这个图片的坐标系
        motion_model: 对齐模型，默认 "homography"（单应性，能处理轻微视角变化）
                      也可以选 "affine"（仿射，只能处理平移、旋转、缩放）
        fast_mode: 是否使用快速模式（多级优化），默认 True
        max_iterations: 最大迭代次数，默认 50（快速模式）或 200（标准模式）
        eps: 精度阈值，默认 1e-4（快速模式）或 1e-6（标准模式）
    
    返回：
        对齐后的关灯图，尺寸和 fixed 一样。
    """
    h, w = fixed.shape[:2]
    
    # 快速模式：多级优化策略
    if fast_mode and min(h, w) > 800:
        # 第一级：在下采样图像上快速计算
        # 计算下采样比例（目标尺寸约800像素）
        scale = 800.0 / min(h, w)
        h_small = int(h * scale)
        w_small = int(w * scale)
        
        # 下采样（使用 AREA 插值，适合缩小）
        fixed_small = cv2.resize(fixed, (w_small, h_small), interpolation=cv2.INTER_AREA)
        moving_small = cv2.resize(moving, (w_small, h_small), interpolation=cv2.INTER_AREA)
        
        # 在小尺寸上计算变换矩阵
        fixed_gray = cv2.cvtColor(fixed_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving_gray = cv2.cvtColor(moving_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        if motion_model == "homography":
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # 快速模式：降低迭代次数和精度要求
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=fixed_gray,
                inputImage=moving_gray,
                warpMatrix=warp_matrix,
                motionType=warp_mode,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error:
            # 如果 ECC 失败，返回原图（不做变换）
            return moving
        
        # 将变换矩阵正确缩放到原图尺寸
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # 单应矩阵缩放：需要正确缩放所有元素
            # 变换：H_original = S_inv * H_small * S
            sx = w / w_small
            sy = h / h_small
            S_inv = np.array([[1/sx, 0, 0], [0, 1/sy, 0], [0, 0, 1]], dtype=np.float32)
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
            warp_matrix = S_inv @ warp_matrix @ S
        else:
            # 仿射矩阵缩放：缩放平移部分
            sx = w / w_small
            sy = h / h_small
            warp_matrix[0, 2] *= sx  # tx
            warp_matrix[1, 2] *= sy  # ty
        
        # 第二级：在原图上进行精细优化（提高精度）
        fixed_gray_full = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving_gray_full = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 精细优化：使用更严格的精度，但迭代次数较少（因为初始值已经很好）
        criteria_refine = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-5)
        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=fixed_gray_full,
                inputImage=moving_gray_full,
                warpMatrix=warp_matrix,
                motionType=warp_mode,
                criteria=criteria_refine,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error:
            # 如果精细优化失败，使用下采样得到的矩阵
            pass
    else:
        # 标准模式：在原图上直接计算
        fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if motion_model == "homography":
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=fixed_gray,
                inputImage=moving_gray,
                warpMatrix=warp_matrix,
                motionType=warp_mode,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error:
            return moving

    # 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        registered = cv2.warpPerspective(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        registered = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return registered

