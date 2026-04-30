"""
平整度检测主流程

按照以下步骤执行：
1. 读取数据（./data/left_env.png、./data/right_env.png、./data/left_mix.png、./data/right_mix.png）
2. 差分投影，得出左右棋盘格图和掩码图
3. 通过角点检测，得出左右棋盘格的角点坐标
4. 生成点云，计算平整度
5. 不平整度可视化(./result/flatness.png)
"""
import os
import sys
import glob
import json
import cv2
import numpy as np
from pathlib import Path

# 将算法模块路径添加到 sys.path，以便使用标准导入
_base_dir = Path(__file__).parent
_base_dir_str = str(_base_dir)
if _base_dir_str not in sys.path:
    sys.path.insert(0, _base_dir_str)

# ========== 导入 projector_reflection_diff 模块 ==========
from projector_reflection_diff.main import process_projection_diff as process_projection_diff_core
from projector_reflection_diff.utils.io_utils import safe_imwrite

# ========== 导入 stereo_corner_matching 模块 ==========
from stereo_corner_matching.utils.io_utils import read_image_grayscale
from stereo_corner_matching.core.corner_detection import (
    draw_corners_visualization,
    find_chessboard_corners,
    refine_corners,
)
from stereo_corner_matching.core.corner_matching import compute_disparities, match_by_relative_coordinates
from stereo_corner_matching.core.image_processing import crop_image_by_mask
from stereo_corner_matching.utils.visualization import visualize_matches

# ========== 导入 pointcloud_gen 模块 ==========
from pointcloud_gen.core.stereo_process import process_stereo_matches
from pointcloud_gen.utils.io_utils import load_uv_json

from config import DEFAULT_CONFIG, FlatnessConfig
from errors import CornerDetectionError
from logging_utils import configure_logging, get_logger


logger = get_logger("detect")


def _save_debug_image(result_dir: str, debug_config, name: str, image: np.ndarray) -> None:
    if not debug_config.enabled:
        return
    debug_dir = os.path.join(result_dir, debug_config.output_dir_name)
    os.makedirs(debug_dir, exist_ok=True)
    safe_imwrite(os.path.join(debug_dir, f"{name}.png"), image)


def process_projection_diff(
    env_path: str,
    mix_path: str,
    output_dir: str,
    side: str,
    config: FlatnessConfig = None,
) -> tuple:
    """
    处理单侧的投影反射差分，生成棋盘格图和掩码图
    
    参数:
        env_path: 环境图路径
        mix_path: 混合图路径
        output_dir: 输出目录
        side: 侧别标识（'left' 或 'right'）
    
    返回:
        (chessboard_img, mask_img) 棋盘格图和掩码图
    """
    config = config or DEFAULT_CONFIG
    logger.info("处理 %s 侧投影反射差分: env=%s mix=%s", side, env_path, mix_path)
    
    # 调用核心处理函数
    chessboard_img, chess_mask = process_projection_diff_core(
        env_path,
        mix_path,
        output_dir,
        config=config.projection,
        debug_prefix=side,
    )
    
    # 保存结果
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
    """
    检测棋盘格角点
    
    参数:
        img_path: 棋盘格图像路径
        mask_path: 掩码图路径
        chessboard_size: 棋盘格尺寸 (行数, 列数)
        output_dir: 输出目录
        side: 侧别标识（'left' 或 'right'）
    
    返回:
        角点坐标数组
    """
    config = config or DEFAULT_CONFIG
    corner_config = config.corner_detection
    logger.info("检测 %s 侧角点", side)
    
    # 读取图像和掩码
    img = read_image_grayscale(img_path)
    mask = read_image_grayscale(mask_path)
    
    # 裁剪图像到棋盘格区域
    img_cropped, mask_cropped, offset, ratio = crop_image_by_mask(
        img,
        mask,
        config=corner_config,
    )
    
    # 检测角点
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
    
    # 调整角点坐标（加上裁剪偏移）
    if offset != (0, 0):
        corners[:, 0, 0] += offset[0]
        corners[:, 0, 1] += offset[1]
    
    # 亚像素精化
    corners = refine_corners(img, corners, config=corner_config)
    _save_debug_image(
        output_dir,
        corner_config.debug,
        f"{side}_detected_corners_visualization",
        draw_corners_visualization(img, corners),
    )
    logger.info("%s 侧检测到 %d 个角点", side, len(corners))
    
    return corners


def main(data_dir: str = None, result_dir: str = None, config: FlatnessConfig = None):
    """主函数：执行完整的平整度检测流程
    
    Args:
        data_dir: 数据目录路径，如果为 None 则使用默认路径
        result_dir: 结果目录路径，如果为 None 则使用默认路径
    """
    config = config or DEFAULT_CONFIG
    configure_logging(config.logging.level)

    # 设置路径
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
    
    # ========== 步骤1：读取数据 ==========
    logger.info("[步骤1] 读取数据")
    
    def find_image_file(data_dir: str, pattern: str) -> str:
        """查找包含指定模式的文件，支持多种格式"""
        exts = [".png", ".jpg", ".jpeg", ".bmp"]
        for ext in exts:
            path = os.path.join(data_dir, f"{pattern}{ext}")
            if os.path.exists(path):
                return path
        # 如果精确匹配失败，尝试模糊匹配
        for ext in exts:
            pattern_with_ext = f"*{pattern}*{ext}"
            matches = glob.glob(os.path.join(data_dir, pattern_with_ext))
            if matches:
                return sorted(matches)[0]  # 返回第一个匹配的文件
        raise FileNotFoundError(f"未找到包含 '{pattern}' 的图片文件（支持格式: {', '.join(exts)}）")
    
    # 自动查找文件
    left_env_path = find_image_file(data_dir, "left_env")
    right_env_path = find_image_file(data_dir, "right_env")
    left_mix_path = find_image_file(data_dir, "left_mix")
    right_mix_path = find_image_file(data_dir, "right_mix")
    
    logger.info("左侧环境图: %s", left_env_path)
    logger.info("右侧环境图: %s", right_env_path)
    logger.info("左侧投影图: %s", left_mix_path)
    logger.info("右侧投影图: %s", right_mix_path)
    
    # ========== 步骤2：差分投影，得出左右棋盘格图和掩码图 ==========
    logger.info("[步骤2] 差分投影，得出左右棋盘格图和掩码图")
    
    # 处理左侧
    left_chessboard, left_mask = process_projection_diff(
        left_env_path, left_mix_path, result_dir, "left", config=config
    )
    
    # 处理右侧
    right_chessboard, right_mask = process_projection_diff(
        right_env_path, right_mix_path, result_dir, "right", config=config
    )
    
    # ========== 步骤3：通过角点检测，得出左右棋盘格的角点坐标 ==========
    logger.info("[步骤3] 通过角点检测，得出左右棋盘格的角点坐标")
    
    chessboard_size = config.corner_detection.chessboard_size
    logger.info("棋盘格参数: %d×%d (内角点数)", chessboard_size[0], chessboard_size[1])
    
    # 检测左侧角点
    left_img_path = os.path.join(result_dir, "left_detect.png")
    left_mask_path = os.path.join(result_dir, "left_mask.png")
    corners_left = detect_corners(left_img_path, left_mask_path, chessboard_size, result_dir, "left", config=config)
    
    # 检测右侧角点
    right_img_path = os.path.join(result_dir, "right_detect.png")
    right_mask_path = os.path.join(result_dir, "right_mask.png")
    corners_right = detect_corners(right_img_path, right_mask_path, chessboard_size, result_dir, "right", config=config)
    
    # 角点匹配
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
        raise RuntimeError("角点匹配未找到任何匹配对！")
    
    # 保存角点坐标
    corners_left_path = os.path.join(result_dir, "corners_left.json")
    corners_right_path = os.path.join(result_dir, "corners_right.json")
    
    left_pts = corners_left_matched.reshape(-1, 2).tolist()
    right_pts = corners_right_matched.reshape(-1, 2).tolist()
    
    with open(corners_left_path, 'w', encoding='utf-8') as f:
        json.dump(left_pts, f, indent=2, ensure_ascii=False)
    with open(corners_right_path, 'w', encoding='utf-8') as f:
        json.dump(right_pts, f, indent=2, ensure_ascii=False)
    
    logger.info("角点坐标已保存: %s, %s", corners_left_path, corners_right_path)

    if config.corner_matching.save_quality_json:
        match_quality_path = os.path.join(result_dir, "match_quality.json")
        with open(match_quality_path, "w", encoding="utf-8") as f:
            json.dump(match_quality, f, indent=2, ensure_ascii=False)
        logger.info("匹配质量指标已保存: %s", match_quality_path)

    if config.corner_matching.debug.enabled:
        try:
            debug_dir = os.path.join(result_dir, config.corner_matching.debug.output_dir_name)
            os.makedirs(debug_dir, exist_ok=True)
            visualize_matches(
                read_image_grayscale(left_img_path),
                read_image_grayscale(right_img_path),
                corners_left_matched,
                corners_right_matched,
                compute_disparities(corners_left_matched, corners_right_matched),
                os.path.join(debug_dir, "matched_corners_visualization.png"),
            )
        except Exception as exc:
            logger.warning("匹配可视化输出失败: %s", exc)
    
    # ========== 步骤4：生成点云，计算平整度 ==========
    logger.info("[步骤4] 生成点云，计算平整度")
    
    # 加载角点坐标
    uv_left = load_uv_json(corners_left_path)
    uv_right = load_uv_json(corners_right_path)
    
    # TODO: 根据真实相机标定填写这些参数
    image_shape = (left_chessboard.shape[0], left_chessboard.shape[1])  # (高度, 宽度)
    K = np.array([[800, 0, image_shape[1]/2], 
                   [0, 800, image_shape[0]/2], 
                   [0, 0, 1]], dtype=float)
    baseline = 0.265  # 基线距离（米）
    
    logger.info("相机参数: K=%s, baseline=%sm", K.tolist(), baseline)
    logger.info("图像尺寸: %s", image_shape)
    
    # 处理立体匹配，生成点云和平整度指标
    flatness_vis_path = os.path.join(result_dir, "flatness.png")
    result = process_stereo_matches(
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
        save_fig_path=flatness_vis_path  
    )
    
    # 保存平整度指标
    metrics_path = os.path.join(result_dir, "flatness_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result["flatness_metrics"], f, indent=2, ensure_ascii=False)
        
    logger.info("平整度指标已保存: %s", metrics_path)
    logger.info("平整度指标: %s", result["flatness_metrics"])
    
    # 保存稀疏点云数据用于前端 3D 展示
    pointcloud_data_path = os.path.join(result_dir, "pointcloud_data.json")
    
    # 准备点云数据
    def _to_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    pc_data = {
        "projected_points": _to_list(result.get("projected_pts", [])),
        "projected_dists": _to_list(result.get("projected_z", [])),  # 与颜色对应的 z'
    }
    
    with open(pointcloud_data_path, 'w', encoding='utf-8') as f:
        json.dump(pc_data, f, ensure_ascii=False)
    
    logger.info("3D点云数据已保存: %s", pointcloud_data_path)
    
    # ========== 步骤5：不平整度可视化 ==========
    logger.info("[步骤5] 不平整度可视化")
    if os.path.exists(flatness_vis_path):
        logger.info("不平整度可视化已保存: %s", flatness_vis_path)
    else:
        logger.warning("未找到平整度可视化文件 %s", flatness_vis_path)
    
    logger.info("=== 处理完成 ===")
    logger.info("所有结果已保存到: %s", result_dir)


if __name__ == "__main__":
    main()
