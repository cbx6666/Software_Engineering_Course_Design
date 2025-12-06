"""
文件 I/O 相关函数
"""
import os
import glob
from typing import Tuple

import cv2
import numpy as np


def read_image_bgr(path: str) -> np.ndarray:
    """
    读取彩色图片（BGR 顺序）。
    
    参数：
        path: 图片文件的完整路径（例如 "data/env.jpg"）
    
    返回：
        一个 numpy 数组，形状是 (高度, 宽度, 3)，表示 BGR 三个颜色通道的像素值。
        每个像素值范围是 0-255（uint8 类型）。
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    
    return img


def safe_imwrite(path: str, image: np.ndarray) -> None:
    """
    保存图片到文件。
    
    参数：
        path: 要保存的文件路径（例如 "result/result_detect.png"）
        image: 要保存的图片数组（numpy 数组，通常是 uint8 类型）
    
    返回：
        无
    """
    if not cv2.imwrite(path, image):
        raise IOError(f"写入图像失败: {path}")


def find_input_images(data_dir: str) -> Tuple[str, str]:
    """
    在 data 目录里查找文件名包含 "env" 和 "mix" 的图片。
    
    两张图片必须按照该命名规则提供，缺少任意一侧即报错，不再使用模糊匹配或排序兜底。
    
    参数：
        data_dir: 数据目录路径
    
    返回：
        (env_path, mix_path) 环境图和混合图的路径
    """
    # 支持的图片格式
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    
    # 扫描目录，收集所有图片文件
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, e)))
    
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到图片")
    
    env_candidates = sorted(
        [f for f in files if "env" in os.path.basename(f).lower()])
    mix_candidates = sorted(
        [f for f in files if "mix" in os.path.basename(f).lower()])
    
    if not env_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'env' 的图片")
    if not mix_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'mix' 的图片")
    
    return env_candidates[0], mix_candidates[0]

