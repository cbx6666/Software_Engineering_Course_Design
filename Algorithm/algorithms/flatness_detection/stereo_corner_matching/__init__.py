"""
双目立体视觉角点匹配模块

从左右两张处理后的棋盘格灰度图中检测角点，按顺序匹配对应角点，并计算视差。
"""
from .utils.io_utils import read_image_grayscale, find_input_images, find_mask_images, save_results
from .core.corner_detection import enhance_chessboard_region, find_chessboard_corners, refine_corners
from .core.corner_matching import sort_corners_to_grid, match_by_relative_coordinates, compute_disparities
from .core.image_processing import detect_chessboard_mask, crop_image_by_mask
from .utils.visualization import visualize_matches

__all__ = [
    'read_image_grayscale',
    'find_input_images',
    'find_mask_images',
    'save_results',
    'enhance_chessboard_region',
    'find_chessboard_corners',
    'refine_corners',
    'sort_corners_to_grid',
    'match_by_relative_coordinates',
    'compute_disparities',
    'detect_chessboard_mask',
    'crop_image_by_mask',
    'visualize_matches',
]

