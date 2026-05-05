"""投影反射差分主流程。"""

import os

import cv2
import numpy as np

from .core.alignment import ecc_register, resize_to_match
from .core.brightness_matching import build_fit_mask, channelwise_compensated_diff
from .core.chessboard_processing import (
    build_chess_mask_from_proj,
    enhance_chessboard_contrast,
    suppress_background_in_mask,
)
from .utils.io_utils import find_input_images, read_image_bgr, safe_imwrite

try:
    from ..config import DEFAULT_CONFIG, ProjectionDiffConfig
    from ..logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, ProjectionDiffConfig
    from logging_utils import get_logger


logger = get_logger("projection")


def _save_debug_image(output_dir: str, config: ProjectionDiffConfig, name: str, image: np.ndarray) -> None:
    """按调试配置保存中间图像。"""
    if not config.debug.enabled:
        return
    debug_dir = os.path.join(output_dir, config.debug.output_dir_name)
    os.makedirs(debug_dir, exist_ok=True)
    safe_imwrite(os.path.join(debug_dir, f"{name}.png"), image)


def process_projection_diff(
    env_path: str,
    mix_path: str,
    output_dir: str,
    config: ProjectionDiffConfig = None,
    debug_prefix: str = "projection",
) -> tuple:
    """执行单组环境图/投影图的差分流程。"""
    config = config or DEFAULT_CONFIG.projection

    img_env = read_image_bgr(env_path)
    img_mix = read_image_bgr(mix_path)
    img_env = resize_to_match(img_env, (img_mix.shape[0], img_mix.shape[1]))

    img_env_aligned = ecc_register(
        img_env,
        img_mix,
        motion_model="homography",
        fast_mode=True,
        max_iterations=config.ecc_max_iterations,
        eps=config.ecc_eps,
        config=config,
    )
    _save_debug_image(output_dir, config, f"{debug_prefix}_aligned_env", img_env_aligned)

    mask_fit = build_fit_mask(img_env_aligned, img_mix, config=config)
    _save_debug_image(output_dir, config, f"{debug_prefix}_mask_fit", mask_fit)

    r_proj_bgr = channelwise_compensated_diff(img_env_aligned, img_mix, mask_fit, config=config)
    _save_debug_image(output_dir, config, f"{debug_prefix}_raw_diff", r_proj_bgr)
    r_proj_gray = cv2.cvtColor(r_proj_bgr, cv2.COLOR_BGR2GRAY)

    chess_mask = build_chess_mask_from_proj(r_proj_gray, config=config)
    _save_debug_image(output_dir, config, f"{debug_prefix}_chess_mask", chess_mask)

    r_proj_hf = suppress_background_in_mask(r_proj_gray, chess_mask, ksize=config.background_ksize)
    chessboard_img = enhance_chessboard_contrast(
        r_proj_hf,
        chess_mask,
        clip_limit=config.contrast_clip_limit,
        tile_grid_size=config.contrast_tile_grid_size,
    )
    _save_debug_image(output_dir, config, f"{debug_prefix}_enhanced_chessboard", chessboard_img)

    logger.info(
        "projection diff complete: fit_mask=%.1f%% chess_mask=%.1f%%",
        np.count_nonzero(mask_fit) / mask_fit.size * 100,
        np.count_nonzero(chess_mask) / chess_mask.size * 100,
    )
    return chessboard_img, chess_mask


def main():
    """使用模块内置 `data` 与 `result` 目录执行演示流程。"""
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    env_path, mix_path = find_input_images(data_dir)
    chessboard_img, chess_mask = process_projection_diff(env_path, mix_path, out_dir)

    detect_out = os.path.join(out_dir, "result_detect.png")
    mask_out = os.path.join(out_dir, "result_mask.png")
    safe_imwrite(detect_out, chessboard_img)
    safe_imwrite(mask_out, chess_mask)

    print("=== 投影反射差分流程完成 ===")
    print(f"棋盘格检测图: {detect_out}")
    print(f"棋盘格掩膜图: {mask_out}")


if __name__ == "__main__":
    main()
