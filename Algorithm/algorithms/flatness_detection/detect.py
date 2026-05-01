"""平整度检测主流程入口。"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

_base_dir = Path(__file__).parent
_base_dir_str = str(_base_dir)
if _base_dir_str not in sys.path:
    sys.path.insert(0, _base_dir_str)

from projector_reflection_diff.main import process_projection_diff as process_projection_diff_core
from projector_reflection_diff.utils.io_utils import safe_imwrite
from stereo_corner_matching.core.corner_detection import (
    draw_corners_visualization,
    find_chessboard_corners,
    refine_corners,
)
from stereo_corner_matching.core.corner_matching import compute_disparities, match_by_relative_coordinates
from stereo_corner_matching.core.image_processing import crop_image_by_mask
from stereo_corner_matching.utils.io_utils import read_image_grayscale
from stereo_corner_matching.utils.visualization import visualize_matches
from pointcloud_gen.core.stereo_process import process_stereo_matches
from pointcloud_gen.utils.io_utils import load_uv_json

try:
    from .config import DEFAULT_CONFIG, FlatnessConfig
    from .errors import CornerDetectionError
    from .logging_utils import configure_logging, get_logger
    from .pipeline_helpers import (
        保存匹配质量,
        保存平整度指标,
        保存点云数据,
        保存角点坐标,
        保存调试图,
        查找输入图片,
    )
except ImportError:
    from config import DEFAULT_CONFIG, FlatnessConfig
    from errors import CornerDetectionError
    from logging_utils import configure_logging, get_logger
    from pipeline_helpers import (
        保存匹配质量,
        保存平整度指标,
        保存点云数据,
        保存角点坐标,
        保存调试图,
        查找输入图片,
    )


logger = get_logger("detect")


def process_projection_diff(
    env_path: str,
    mix_path: str,
    output_dir: str,
    side: str,
    config: FlatnessConfig = None,
) -> tuple:
    """处理单侧投影差分，并保存兼容旧流程的输出文件。"""
    config = config or DEFAULT_CONFIG
    logger.info("处理 %s 侧投影反射差分: env=%s mix=%s", side, env_path, mix_path)

    chessboard_img, chess_mask = process_projection_diff_core(
        env_path,
        mix_path,
        output_dir,
        config=config.projection,
        debug_prefix=side,
    )

    detect_out = os.path.join(output_dir, f"{side}_detect.png")
    mask_out = os.path.join(output_dir, f"{side}_mask.png")
    safe_imwrite(detect_out, chessboard_img)
    safe_imwrite(mask_out, chess_mask)
    logger.info("保存 %s 侧投影差分结果: %s, %s", side, detect_out, mask_out)
    return chessboard_img, chess_mask


def detect_corners(
    img_path: str,
    mask_path: str,
    chessboard_size: tuple,
    output_dir: str,
    side: str,
    config: FlatnessConfig = None,
) -> np.ndarray:
    """检测单侧棋盘格角点，并输出调试可视化。"""
    config = config or DEFAULT_CONFIG
    corner_config = config.corner_detection
    logger.info("检测 %s 侧角点", side)

    img = read_image_grayscale(img_path)
    mask = read_image_grayscale(mask_path)
    img_cropped, mask_cropped, offset, _ = crop_image_by_mask(img, mask, config=corner_config)

    try:
        corners = find_chessboard_corners(
            img_cropped,
            chessboard_size,
            allow_partial=corner_config.allow_partial,
            enhance=False,
            mask=mask_cropped,
            config=corner_config,
            raise_on_failure=True,
        )
    except CornerDetectionError as exc:
        raise CornerDetectionError(f"{side} 侧角点检测失败：{exc}") from exc

    if offset != (0, 0):
        corners[:, 0, 0] += offset[0]
        corners[:, 0, 1] += offset[1]

    corners = refine_corners(img, corners, config=corner_config)
    保存调试图(
        output_dir,
        corner_config.debug,
        f"{side}_detected_corners_visualization",
        draw_corners_visualization(img, corners),
    )
    logger.info("%s 侧检测到 %d 个角点", side, len(corners))
    return corners


def _匹配左右角点(
    corners_left: np.ndarray,
    corners_right: np.ndarray,
    chessboard_size: tuple,
    result_dir: str,
    config: FlatnessConfig,
) -> tuple:
    """完成左右角点匹配，并按配置输出调试信息。"""
    logger.info("进行角点匹配...")
    corners_left_matched, corners_right_matched, match_quality = match_by_relative_coordinates(
        corners_left,
        corners_right,
        chessboard_size,
        config=config.corner_matching,
        return_quality=True,
    )
    logger.info("匹配成功: %d 对", len(corners_left_matched))

    if len(corners_left_matched) == 0:
        raise RuntimeError("角点匹配未找到任何匹配对")

    if config.corner_matching.save_quality_json:
        match_quality_path = 保存匹配质量(result_dir, match_quality)
        logger.info("匹配质量指标已保存: %s", match_quality_path)

    if config.corner_matching.debug.enabled:
        try:
            debug_dir = os.path.join(result_dir, config.corner_matching.debug.output_dir_name)
            os.makedirs(debug_dir, exist_ok=True)
            visualize_matches(
                read_image_grayscale(os.path.join(result_dir, "left_detect.png")),
                read_image_grayscale(os.path.join(result_dir, "right_detect.png")),
                corners_left_matched,
                corners_right_matched,
                compute_disparities(corners_left_matched, corners_right_matched),
                os.path.join(debug_dir, "matched_corners_visualization.png"),
            )
        except Exception as exc:
            logger.warning("匹配可视化输出失败: %s", exc)

    return corners_left_matched, corners_right_matched


def _执行点云与平整度计算(
    result_dir: str,
    left_chessboard: np.ndarray,
) -> dict:
    """加载匹配结果并执行点云重建与平整度计算。"""
    corners_left_path = os.path.join(result_dir, "corners_left.json")
    corners_right_path = os.path.join(result_dir, "corners_right.json")
    uv_left = load_uv_json(corners_left_path)
    uv_right = load_uv_json(corners_right_path)

    image_shape = (left_chessboard.shape[0], left_chessboard.shape[1])
    K = np.array(
        [
            [800, 0, image_shape[1] / 2],
            [0, 800, image_shape[0] / 2],
            [0, 0, 1],
        ],
        dtype=float,
    )
    baseline = 0.265

    logger.info("相机参数: K=%s, baseline=%sm", K.tolist(), baseline)
    logger.info("图像尺寸: %s", image_shape)

    return process_stereo_matches(
        uv_left,
        uv_right,
        K,
        baseline,
        image_shape=image_shape,
        densify=True,
        densify_method="cubic",
        densify_smooth_sigma=1.0,
        mad_thresh=3.5,
        export_ply_path=None,
        export_csv_path=None,
        visualize=True,
        save_fig_path=os.path.join(result_dir, "flatness.png"),
    )


def main(data_dir: str = None, result_dir: str = None, config: FlatnessConfig = None):
    """执行完整的平整度检测流程。"""
    config = config or DEFAULT_CONFIG
    configure_logging(config.logging.level)

    if data_dir is None or result_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if data_dir is None:
            data_dir = os.path.join(base_dir, "data")
        if result_dir is None:
            result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    logger.info("=== 平整度检测流程开始 ===")
    logger.info("数据目录: %s", data_dir)
    logger.info("结果目录: %s", result_dir)

    logger.info("[步骤1] 读取数据")
    left_env_path = 查找输入图片(data_dir, "left_env")
    right_env_path = 查找输入图片(data_dir, "right_env")
    left_mix_path = 查找输入图片(data_dir, "left_mix")
    right_mix_path = 查找输入图片(data_dir, "right_mix")
    logger.info("左侧环境图: %s", left_env_path)
    logger.info("右侧环境图: %s", right_env_path)
    logger.info("左侧投影图: %s", left_mix_path)
    logger.info("右侧投影图: %s", right_mix_path)

    logger.info("[步骤2] 差分投影，得出左右棋盘格图和掩码图")
    left_chessboard, _ = process_projection_diff(left_env_path, left_mix_path, result_dir, "left", config=config)
    right_chessboard, _ = process_projection_diff(right_env_path, right_mix_path, result_dir, "right", config=config)

    logger.info("[步骤3] 通过角点检测，得出左右棋盘格的角点坐标")
    chessboard_size = config.corner_detection.chessboard_size
    logger.info("棋盘格参数: %d×%d (内角点数)", chessboard_size[0], chessboard_size[1])
    corners_left = detect_corners(
        os.path.join(result_dir, "left_detect.png"),
        os.path.join(result_dir, "left_mask.png"),
        chessboard_size,
        result_dir,
        "left",
        config=config,
    )
    corners_right = detect_corners(
        os.path.join(result_dir, "right_detect.png"),
        os.path.join(result_dir, "right_mask.png"),
        chessboard_size,
        result_dir,
        "right",
        config=config,
    )

    corners_left_matched, corners_right_matched = _匹配左右角点(
        corners_left,
        corners_right,
        chessboard_size,
        result_dir,
        config,
    )

    saved_paths = 保存角点坐标(
        result_dir,
        corners_left_matched.reshape(-1, 2).tolist(),
        corners_right_matched.reshape(-1, 2).tolist(),
    )
    logger.info("角点坐标已保存: %s, %s", saved_paths["left"], saved_paths["right"])

    logger.info("[步骤4] 生成点云，计算平整度")
    result = _执行点云与平整度计算(result_dir, left_chessboard)
    metrics_path = 保存平整度指标(result_dir, result["flatness_metrics"])
    logger.info("平整度指标已保存: %s", metrics_path)
    logger.info("平整度指标: %s", result["flatness_metrics"])

    pointcloud_data_path = 保存点云数据(result_dir, result)
    logger.info("3D点云数据已保存: %s", pointcloud_data_path)

    logger.info("[步骤5] 不平整度可视化")
    flatness_vis_path = os.path.join(result_dir, "flatness.png")
    if os.path.exists(flatness_vis_path):
        logger.info("不平整度可视化已保存: %s", flatness_vis_path)
    else:
        logger.warning("未找到平整度可视化文件 %s", flatness_vis_path)

    logger.info("=== 处理完成 ===")
    logger.info("所有结果已保存到: %s", result_dir)


if __name__ == "__main__":
    main()
