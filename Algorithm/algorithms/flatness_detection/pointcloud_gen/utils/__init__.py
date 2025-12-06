"""
工具函数模块

提供相机模型、插值、平面拟合、异常值检测、I/O 等工具函数。
"""
from .camera import depth_from_pixels, backproject_uv_to_xyz
from .interp import densify_disparity
from .outliers import mad_mask
from .plane_fit import fit_plane_least_squares, point_plane_signed_distance
from .io_utils import load_uv_json, save_ply, export_csv, visualize_pointcloud

__all__ = [
    'depth_from_pixels',
    'backproject_uv_to_xyz',
    'densify_disparity',
    'mad_mask',
    'fit_plane_least_squares',
    'point_plane_signed_distance',
    'load_uv_json',
    'save_ply',
    'export_csv',
    'visualize_pointcloud',
]

