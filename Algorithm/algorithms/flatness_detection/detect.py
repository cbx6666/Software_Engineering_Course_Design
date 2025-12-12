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
from stereo_corner_matching.core.corner_detection import find_chessboard_corners, refine_corners
from stereo_corner_matching.core.corner_matching import match_by_relative_coordinates
from stereo_corner_matching.core.image_processing import crop_image_by_mask

# ========== 导入 pointcloud_gen 模块 ==========
from pointcloud_gen.core.stereo_process import process_stereo_matches
from pointcloud_gen.utils.io_utils import load_uv_json


def process_projection_diff(env_path: str, mix_path: str, output_dir: str, side: str) -> tuple:
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
    print(f"\n=== 处理 {side} 侧投影反射差分 ===")
    print(f"读取图片: {env_path}, {mix_path}")
    
    # 调用核心处理函数
    chessboard_img, chess_mask = process_projection_diff_core(env_path, mix_path, output_dir)
    
    # 保存结果
    detect_out = os.path.join(output_dir, f"{side}_detect.png")
    mask_out = os.path.join(output_dir, f"{side}_mask.png")
    safe_imwrite(detect_out, chessboard_img)
    safe_imwrite(mask_out, chess_mask)
    print(f"保存结果: {detect_out}, {mask_out}")
    
    return chessboard_img, chess_mask


def detect_corners(img_path: str, mask_path: str, chessboard_size: tuple, output_dir: str, side: str) -> np.ndarray:
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
    print(f"\n=== 检测 {side} 侧角点 ===")
    
    # 读取图像和掩码
    img = read_image_grayscale(img_path)
    mask = read_image_grayscale(mask_path)
    
    # 裁剪图像到棋盘格区域
    img_cropped, mask_cropped, offset, ratio = crop_image_by_mask(img, mask, padding=50)
    
    # 检测角点
    corners = find_chessboard_corners(
        img_cropped, 
        chessboard_size, 
        allow_partial=True, 
        enhance=False,
        mask=mask_cropped
    )
    
    if corners is None:
        raise RuntimeError(f"{side} 侧角点检测失败！")
    
    # 调整角点坐标（加上裁剪偏移）
    if offset != (0, 0):
        corners[:, 0, 0] += offset[0]
        corners[:, 0, 1] += offset[1]
    
    # 亚像素精化
    corners = refine_corners(img, corners)
    
    print(f"{side} 侧检测到 {len(corners)} 个角点")
    
    return corners


def main():
    """主函数：执行完整的平整度检测流程"""
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== 平整度检测流程开始 ===")
    print(f"数据目录: {data_dir}")
    print(f"结果目录: {result_dir}")
    
    # ========== 步骤1：读取数据 ==========
    print("\n[步骤1] 读取数据")
    
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
    
    print(f"  左侧环境图: {left_env_path}")
    print(f"  右侧环境图: {right_env_path}")
    print(f"  左侧投影图: {left_mix_path}")
    print(f"  右侧投影图: {right_mix_path}")
    
    # ========== 步骤2：差分投影，得出左右棋盘格图和掩码图 ==========
    print("\n[步骤2] 差分投影，得出左右棋盘格图和掩码图")
    
    # 处理左侧
    left_chessboard, left_mask = process_projection_diff(
        left_env_path, left_mix_path, result_dir, "left"
    )
    
    # 处理右侧
    right_chessboard, right_mask = process_projection_diff(
        right_env_path, right_mix_path, result_dir, "right"
    )
    
    # ========== 步骤3：通过角点检测，得出左右棋盘格的角点坐标 ==========
    print("\n[步骤3] 通过角点检测，得出左右棋盘格的角点坐标")
    
    chessboard_size = (7, 10)  # 棋盘格内角点数 (行数, 列数)
    print(f"棋盘格参数: {chessboard_size[0]}×{chessboard_size[1]} (内角点数)")
    
    # 检测左侧角点
    left_img_path = os.path.join(result_dir, "left_detect.png")
    left_mask_path = os.path.join(result_dir, "left_mask.png")
    corners_left = detect_corners(left_img_path, left_mask_path, chessboard_size, result_dir, "left")
    
    # 检测右侧角点
    right_img_path = os.path.join(result_dir, "right_detect.png")
    right_mask_path = os.path.join(result_dir, "right_mask.png")
    corners_right = detect_corners(right_img_path, right_mask_path, chessboard_size, result_dir, "right")
    
    # 角点匹配
    print("\n进行角点匹配...")
    corners_left_matched, corners_right_matched = match_by_relative_coordinates(
        corners_left, corners_right, chessboard_size
    )
    
    print(f"匹配成功: {len(corners_left_matched)} 对")
    
    if len(corners_left_matched) == 0:
        raise RuntimeError("角点匹配未找到任何匹配对！")
    
    # 保存角点坐标
    import json
    corners_left_path = os.path.join(result_dir, "corners_left.json")
    corners_right_path = os.path.join(result_dir, "corners_right.json")
    
    left_pts = corners_left_matched.reshape(-1, 2).tolist()
    right_pts = corners_right_matched.reshape(-1, 2).tolist()
    
    with open(corners_left_path, 'w', encoding='utf-8') as f:
        json.dump(left_pts, f, indent=2, ensure_ascii=False)
    with open(corners_right_path, 'w', encoding='utf-8') as f:
        json.dump(right_pts, f, indent=2, ensure_ascii=False)
    
    print(f"角点坐标已保存: {corners_left_path}, {corners_right_path}")
    
    # ========== 步骤4：生成点云，计算平整度 ==========
    print("\n[步骤4] 生成点云，计算平整度")
    
    # 加载角点坐标
    uv_left = load_uv_json(corners_left_path)
    uv_right = load_uv_json(corners_right_path)
    
    # TODO: 根据真实相机标定填写这些参数
    image_shape = (left_chessboard.shape[0], left_chessboard.shape[1])  # (高度, 宽度)
    K = np.array([[800, 0, image_shape[1]/2], 
                  [0, 800, image_shape[0]/2], 
                  [0, 0, 1]], dtype=float)
    baseline = 0.25  # 基线距离（米）
    
    print(f"相机参数: K={K.tolist()}, baseline={baseline}m")
    print(f"图像尺寸: {image_shape}")
    
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
    import json
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result["flatness_metrics"], f, indent=2, ensure_ascii=False)
    
    print(f"平整度指标已保存: {metrics_path}")
    print("平整度指标:", result["flatness_metrics"])
    
    # 保存点云数据用于前端 3D 展示（统一使用投影数据）
    pointcloud_data_path = os.path.join(result_dir, "pointcloud_data.json")
    
    # 准备点云数据
    def _to_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    # 确保投影数据存在
    if "projected_pts" not in result or "projected_z" not in result:
        raise RuntimeError("缺少投影数据：projected_pts 和 projected_z 必须存在")
    
    # 统一使用投影数据，只保留必需字段
    pc_data = {
        "projected_points": _to_list(result["projected_pts"]),  # 投影后的点坐标 (meters)
        "projected_dists": _to_list(result["projected_z"]),      # 投影后的 Z' 值，用于颜色映射 (meters)
    }
    
    with open(pointcloud_data_path, 'w', encoding='utf-8') as f:
        json.dump(pc_data, f, ensure_ascii=False)
    
    print(f"3D点云数据已保存: {pointcloud_data_path}")
    print(f"  投影点数: {len(pc_data['projected_points'])}")
    
    # ========== 步骤5：不平整度可视化 ==========
    print("\n[步骤5] 不平整度可视化")
    if os.path.exists(flatness_vis_path):
        print(f"不平整度可视化已保存: {flatness_vis_path}")
    else:
        print(f"警告: 未找到平整度可视化文件 {flatness_vis_path}")
    
    print("\n=== 处理完成 ===")
    print(f"所有结果已保存到: {result_dir}")


if __name__ == "__main__":
    main()