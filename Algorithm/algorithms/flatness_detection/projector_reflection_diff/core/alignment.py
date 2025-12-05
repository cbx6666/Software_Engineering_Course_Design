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


def ecc_register(moving: np.ndarray, fixed: np.ndarray, motion_model: str = "homography") -> np.ndarray:
    """
    把关灯图的位置"对齐"到开灯图，消除相机轻微移动造成的错位。
    
    即使拍摄时相机有轻微移动或角度变化，也能让两张图片的像素点"对得上"，这样后续相减时不会出现重影或模糊。
    无人机或手持拍摄时，很难保证两张图片完全对齐。如果直接相减，错位的地方会出现"重影"，影响结果质量。
    对齐后，同一位置的像素才真正对应同一块玻璃区域。
    
    原理：
        1. 将两张图片转换为灰度图并归一化到 [0, 1] 范围（ECC 算法要求）
        2. 根据运动模型选择变换类型（单应性或仿射）和初始变换矩阵
        3. 使用 ECC（增强相关系数）算法，通过最大化两张图的相似度来找到最佳变换矩阵
        4. 迭代优化变换参数，直到对齐精度达到要求（最大迭代次数 200，精度阈值 1e-6）
        5. 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
        6. 如果 ECC 失败（比如两张图差异太大），返回原图不做处理
    
    参数：
        moving: 需要被对齐的图片（关灯图）
        fixed:  参考图片（开灯图），moving 会被对齐到这个图片的坐标系
        motion_model: 对齐模型，默认 "homography"（单应性，能处理轻微视角变化）
                      也可以选 "affine"（仿射，只能处理平移、旋转、缩放）
    
    返回：
        对齐后的关灯图，尺寸和 fixed 一样。
    """
    # 转换为灰度图并归一化到 [0, 1] 范围（ECC 算法要求）
    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 根据运动模型选择变换类型和初始变换矩阵
    if motion_model == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)  # 3x3 单应矩阵
    else:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)  # 2x3 仿射矩阵

    # 设置迭代终止条件：最大迭代次数 200，精度阈值 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    
    # 使用 ECC 算法计算最优变换矩阵
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

    # 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
    h, w = fixed.shape[:2]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # 单应变换（透视变换）
        registered = cv2.warpPerspective(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # 仿射变换
        registered = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return registered

