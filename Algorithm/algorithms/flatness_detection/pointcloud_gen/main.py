"""
点云生成和平整度计算主程序

从左右两张图像的角点坐标生成点云，并计算平整度指标。

输入要求：
1) 角点坐标文件（放在 data 文件夹下）：
   - 左图角点：corners_left.json
   - 右图角点：corners_right.json

2) 相机标定参数（在 main() 函数中设置）：
   - image_shape: 图像尺寸 (高度, 宽度)
   - K: 相机内参矩阵 (3×3)
   - baseline: 基线距离（米）

输出文件（保存到 result 文件夹）：
- flatness_metrics.json：平整度指标
- pointcloud.png：点云可视化图像

处理流程：
1) 读取左右角点坐标
2) 从角点坐标恢复 3D 点云
3) 平面拟合和距离计算
4) 视差稠密化（可选）
5) 计算平整度指标
6) 保存结果和可视化图像
"""
import os
import json

import numpy as np

from .core.stereo_process import process_stereo_matches
from .utils.io_utils import load_uv_json


def main():
    """
    主函数：执行完整的点云生成和平整度计算流程。
    
    从左右两张图像的角点坐标生成点云，拟合平面，计算平整度指标。
    """
    # ========== 初始化：设置输入输出目录 ==========
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=== 点云生成和平整度计算流程启动 ===")
    print(f"工作目录: {proj_root}")
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {out_dir}")
    
    # ========== 读取角点坐标 ==========
    print("\n[步骤1] 读取角点坐标")
    uv_left_path = os.path.join(data_dir, "corners_left.json")
    uv_right_path = os.path.join(data_dir, "corners_right.json")
    
    if not os.path.exists(uv_left_path):
        raise FileNotFoundError(f"未找到左图角点文件: {uv_left_path}")
    if not os.path.exists(uv_right_path):
        raise FileNotFoundError(f"未找到右图角点文件: {uv_right_path}")
    
    uv_left_sparse = load_uv_json(uv_left_path)
    uv_right_sparse = load_uv_json(uv_right_path)
    
    print(f"  左图角点: {uv_left_path} ({len(uv_left_sparse)} 个点)")
    print(f"  右图角点: {uv_right_path} ({len(uv_right_sparse)} 个点)")
    
    # ========== 设置相机参数 ==========
    print("\n[步骤2] 设置相机参数")
    # TODO: 根据真实相机标定填写这些参数
    image_shape = (480, 640)  # (高度, 宽度)
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)
    baseline = 0.11  # 基线距离（米）
    
    print(f"  图像尺寸: {image_shape}")
    print(f"  相机内参 K:\n{K}")
    print(f"  基线距离: {baseline} m")
    print("  [提示] 请根据真实相机标定修改这些参数")
    
    # ========== 处理立体匹配，生成点云和平整度指标 ==========
    print("\n[步骤3] 处理立体匹配，生成点云和平整度指标")
    save_fig_path = os.path.join(out_dir, "pointcloud.png")
    
    result = process_stereo_matches(
        uv_left_sparse,
        uv_right_sparse,
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
        save_fig_path=save_fig_path,
    )
    
    # ========== 保存平整度指标 ==========
    print("\n[步骤4] 保存平整度指标")
    metrics_path = os.path.join(out_dir, "flatness_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result["flatness_metrics"], f, indent=2, ensure_ascii=False)
    
    print(f"  平整度指标已保存: {metrics_path}")
    print("\n平整度指标:")
    for key, value in result["flatness_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n[完成] 文件已保存：")
    print(f"  平整度指标: {metrics_path}")
    if os.path.exists(save_fig_path):
        print(f"  点云可视化: {save_fig_path}")
    print("=== 处理结束 ===")

if __name__ == "__main__":
    main()
