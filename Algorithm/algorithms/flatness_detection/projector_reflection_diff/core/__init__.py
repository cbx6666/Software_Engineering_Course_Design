"""
投影反射差分核心模块

包含图像对齐、亮度匹配和棋盘格处理等核心算法。
"""
from .alignment import resize_to_match, ecc_register
from .brightness_matching import build_fit_mask, channelwise_compensated_diff
from .chessboard_processing import (
    build_chess_mask_from_proj,
    suppress_background_in_mask,
    enhance_chessboard_contrast
)

__all__ = [
    'resize_to_match',
    'ecc_register',
    'build_fit_mask',
    'channelwise_compensated_diff',
    'build_chess_mask_from_proj',
    'suppress_background_in_mask',
    'enhance_chessboard_contrast',
]
