"""投影反射差分子模块。"""

from .core.alignment import ecc_register, resize_to_match
from .core.brightness_matching import build_fit_mask, channelwise_compensated_diff
from .core.chessboard_processing import (
    build_chess_mask_from_proj,
    enhance_chessboard_contrast,
    suppress_background_in_mask,
)
from .main import process_projection_diff
from .utils.io_utils import find_input_images, read_image_bgr, safe_imwrite

__all__ = [
    "read_image_bgr",
    "safe_imwrite",
    "find_input_images",
    "resize_to_match",
    "ecc_register",
    "build_fit_mask",
    "channelwise_compensated_diff",
    "build_chess_mask_from_proj",
    "suppress_background_in_mask",
    "enhance_chessboard_contrast",
    "process_projection_diff",
]
