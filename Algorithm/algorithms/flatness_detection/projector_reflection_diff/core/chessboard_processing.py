"""
棋盘格处理相关函数
"""
from typing import Tuple
import cv2
import numpy as np


def build_chess_mask_from_proj(r_proj_gray: np.ndarray, block_size: int = 31, C: int = -5, min_area: int = 200) -> np.ndarray:
    """
    根据"纯投影反射"的灰度图，自动找出棋盘格所在的大致区域。
    
    生成一个掩膜，标记出棋盘格覆盖的区域。这样后续处理时，只在棋盘格区域内做背景抑制，不会影响其他区域。
    棋盘格区域里有缓慢变化的背景反射（比如天空、建筑），会影响角点检测。但非棋盘格区域不需要处理，
    所以先找出棋盘格在哪里，只处理那一块。
    
    原理：
        1. 先用高斯模糊去噪，让图片更平滑
        2. 用自适应阈值把图片二值化（分成黑白两部分），棋盘格的亮暗条纹会很明显
        3. 用形态学闭运算（先膨胀后腐蚀）填补小孔洞，让棋盘格区域更完整
        4. 找出所有连通域，删除面积太小的（这些是噪声）
        5. 适度膨胀掩膜，确保覆盖到方格内部，不会漏掉边缘
    
    参数：
        r_proj_gray: 纯投影反射的灰度图（已经做过对齐和相减）
        block_size: 自适应阈值的块大小，默认 31（必须是奇数）
        C: 阈值常数，默认 -5（负值更容易保留亮暗条纹）
        min_area: 最小连通域面积，默认 200（小于这个面积的区域会被当成噪声删除）
    
    返回：
        一个二值掩膜（uint8），255 表示"这里是棋盘格区域"，0 表示"不是"。
    """
    # 高斯模糊去噪，使图像更平滑
    blur = cv2.GaussianBlur(r_proj_gray, (5, 5), 0)
    
    # 自适应阈值二值化：将图像分为黑白两部分，突出棋盘格的亮暗条纹
    # block_size 必须是奇数，使用按位或确保为奇数
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, block_size | 1), C)
    
    # 形态学闭运算：先膨胀后腐蚀，填补小孔洞，使棋盘格区域更完整
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 连通域分析：找出所有独立的区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    # 筛选出面积大于阈值的连通域（去除小噪声）
    mask = np.zeros_like(bw)
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    
    # 适度膨胀掩膜，确保覆盖到方格内部，不会漏掉边缘
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel2, iterations=1)
    
    return mask


def suppress_background_in_mask(r_proj_gray: np.ndarray, mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    在棋盘格区域内，去掉缓慢变化的背景亮度，只保留细节纹理，让角点更突出。
    
    使用"高斯高通滤波"，把图片分解成"低频（缓慢变化）"和"高频（细节纹理）"，然后去掉低频部分，只保留高频。
    这样棋盘格的角点会更清晰，而背景反射会被抑制。棋盘格区域里除了投影图案，还有缓慢变化的背景反射（比如天空、建筑倒影）。
    这些背景会干扰角点检测。通过高通滤波，把缓慢变化的背景去掉，只保留棋盘格的快速变化（黑白交替），角点就会更明显。
    
    原理：
        1. 确保核大小为奇数（高斯模糊要求）
        2. 用大核高斯模糊得到"低频分量"（缓慢变化的背景）
        3. 用原图减去低频分量，得到"高频分量"（细节纹理，包括棋盘格）
        4. 只在掩膜标记的区域内用高频分量替换原图，其他区域保持原样
        5. 把结果限制在 0-255 范围内
    
    参数：
        r_proj_gray: 纯投影反射的灰度图
        mask: 棋盘格掩膜（255=棋盘格区域，0=非棋盘格区域）
        ksize: 高斯模糊的核大小，默认 51（必须是奇数）。越大，去掉的"低频"越多，
               但太大可能会影响角点附近的细节。
    
    返回：
        处理后的灰度图。只在掩膜标记的区域内做了背景抑制，其他区域保持原样。
    """
    # 确保核大小为奇数（高斯模糊要求）
    ksize = max(3, ksize | 1)
    
    # 高斯模糊得到低频分量（缓慢变化的背景）
    low = cv2.GaussianBlur(r_proj_gray, (ksize, ksize), 0)
    
    # 原图减去低频分量，得到高频分量（细节纹理）
    filtered = cv2.subtract(r_proj_gray, low)
    
    # 复制原图作为输出，使用 int16 类型以支持负值
    out = r_proj_gray.copy().astype(np.int16)
    
    # 仅在掩膜标记的区域内用高频分量替换原图
    out[mask > 0] = filtered[mask > 0].astype(np.int16)
    
    # 限制像素值在 [0, 255] 范围内，并转换为 uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_chessboard_contrast(img: np.ndarray, mask: np.ndarray, 
                                clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    在棋盘格区域内增强对比度，使角点更突出。
    
    使用 CLAHE（对比度受限的自适应直方图均衡化）算法，在棋盘格区域内增强对比度，
    使黑白交替的棋盘格图案更加清晰，便于后续的角点检测。
    
    原理：
        1. 创建 CLAHE 对象，设置对比度限制和网格大小
        2. 对整个图像应用 CLAHE（即使只增强掩膜区域，也需要处理全图以保持一致性）
        3. 只在掩膜标记的区域内用增强后的图像替换原图，其他区域保持原样
        4. 这样可以避免非棋盘格区域的过度增强，同时让棋盘格区域的对比度更明显
    
    参数：
        img: 输入灰度图
        mask: 棋盘格掩膜（255=棋盘格区域，0=非棋盘格区域）
        clip_limit: CLAHE 的对比度限制，默认 2.5。值越大，对比度增强越强，但可能产生过度增强
        tile_grid_size: CLAHE 的网格大小，默认 (8, 8)。将图像分成 8×8 的网格，对每个网格独立进行直方图均衡化
    
    返回：
        处理后的灰度图。只在掩膜标记的区域内做了对比度增强，其他区域保持原样。
    """
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # 对整个图像应用 CLAHE
    enhanced = clahe.apply(img)
    
    # 复制原图作为输出
    out = img.copy()
    
    # 仅在掩膜标记的区域内用增强后的图像替换原图
    out[mask > 0] = enhanced[mask > 0]
    
    return out

