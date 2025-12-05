"""
点云生成和平整度计算模块

从左右两张图像的角点坐标生成点云，并计算平整度指标。

主要功能：
- 从角点坐标恢复 3D 点云
- 平面拟合和距离计算
- 视差稠密化
- 平整度指标计算
- 点云可视化和导出
"""
from .core.stereo_process import process_stereo_matches
from .utils.io_utils import load_uv_json

__all__ = [
    'process_stereo_matches',
    'load_uv_json',
]

