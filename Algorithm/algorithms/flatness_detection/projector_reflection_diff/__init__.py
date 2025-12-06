"""
投影反射差分模块

从两张同一位置拍摄的图片中，提取"投影灯打开时"才出现的那部分反射图案，
让棋盘格等投影图案在图片里更清楚，便于后续角点检测与三维重建。
"""
from .utils.io_utils import read_image_bgr, safe_imwrite, find_input_images
from .core.alignment import resize_to_match, ecc_register
from .core.brightness_matching import build_fit_mask, channelwise_compensated_diff
from .core.chessboard_processing import (
    build_chess_mask_from_proj,
    suppress_background_in_mask,
    enhance_chessboard_contrast
)
from .main import process_projection_diff

__all__ = [
    'read_image_bgr',
    'safe_imwrite',
    'find_input_images',
    'resize_to_match',
    'ecc_register',
    'build_fit_mask',
    'channelwise_compensated_diff',
    'build_chess_mask_from_proj',
    'suppress_background_in_mask',
    'enhance_chessboard_contrast',
    'process_projection_diff',
]

