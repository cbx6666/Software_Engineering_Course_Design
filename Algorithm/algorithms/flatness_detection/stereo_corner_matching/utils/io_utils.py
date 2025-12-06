"""
文件 I/O 相关函数
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

