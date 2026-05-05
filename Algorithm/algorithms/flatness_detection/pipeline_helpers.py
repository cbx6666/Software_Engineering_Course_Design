"""平整度检测主流程辅助函数。"""

import glob
import json
import os
from typing import Any, Dict

import numpy as np

try:
    from .projector_reflection_diff.utils.io_utils import safe_imwrite
except ImportError:
    from projector_reflection_diff.utils.io_utils import safe_imwrite


def 保存调试图(result_dir: str, debug_config, name: str, image: np.ndarray) -> None:
    """按调试配置保存中间图。"""
    if not debug_config.enabled:
        return
    debug_dir = os.path.join(result_dir, debug_config.output_dir_name)
    os.makedirs(debug_dir, exist_ok=True)
    safe_imwrite(os.path.join(debug_dir, f"{name}.png"), image)


def 查找输入图片(data_dir: str, pattern: str) -> str:
    """按约定命名查找输入图片，支持常见扩展名和模糊匹配。"""
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    for ext in exts:
        path = os.path.join(data_dir, f"{pattern}{ext}")
        if os.path.exists(path):
            return path

    for ext in exts:
        fuzzy_pattern = os.path.join(data_dir, f"*{pattern}*{ext}")
        matches = glob.glob(fuzzy_pattern)
        if matches:
            return sorted(matches)[0]

    raise FileNotFoundError(
        f"未找到包含 '{pattern}' 的图片文件（支持格式：{', '.join(exts)}）"
    )


def 保存角点坐标(result_dir: str, left_pts, right_pts) -> Dict[str, str]:
    """保存左右角点坐标 JSON。"""
    corners_left_path = os.path.join(result_dir, "corners_left.json")
    corners_right_path = os.path.join(result_dir, "corners_right.json")

    with open(corners_left_path, "w", encoding="utf-8") as f:
        json.dump(left_pts, f, indent=2, ensure_ascii=False)
    with open(corners_right_path, "w", encoding="utf-8") as f:
        json.dump(right_pts, f, indent=2, ensure_ascii=False)

    return {"left": corners_left_path, "right": corners_right_path}


def 保存匹配质量(result_dir: str, match_quality: Dict[str, Any]) -> str:
    """保存角点匹配质量指标。"""
    match_quality_path = os.path.join(result_dir, "match_quality.json")
    with open(match_quality_path, "w", encoding="utf-8") as f:
        json.dump(match_quality, f, indent=2, ensure_ascii=False)
    return match_quality_path


def 保存平整度指标(result_dir: str, metrics: Dict[str, Any]) -> str:
    """保存平整度指标 JSON。"""
    metrics_path = os.path.join(result_dir, "flatness_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics_path


def 保存点云数据(result_dir: str, result: Dict[str, Any]) -> str:
    """保存前端展示所需的点云数据。"""
    pointcloud_data_path = os.path.join(result_dir, "pointcloud_data.json")

    def 转列表(value):
        return value.tolist() if hasattr(value, "tolist") else list(value)

    pc_data = {
        "projected_points": 转列表(result.get("projected_pts", [])),
        "projected_dists": 转列表(result.get("projected_z", [])),
    }

    with open(pointcloud_data_path, "w", encoding="utf-8") as f:
        json.dump(pc_data, f, ensure_ascii=False)
    return pointcloud_data_path
