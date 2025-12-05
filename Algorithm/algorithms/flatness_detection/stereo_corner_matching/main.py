"""
双目立体视觉角点匹配主程序

从左右两张处理后的棋盘格灰度图中检测角点，按顺序匹配对应角点，并计算视差。

输入要求：
1) 棋盘格灰度图（放在 data 文件夹下）：
   - 左图：文件名包含 "left"（如 left.png）
   - 右图：文件名包含 "right"（如 right.png）

2) 掩膜文件（必须，放在 data 文件夹下）：
   - 左图掩膜：left_mask.png 
   - 右图掩膜：right_mask.png 

3) 棋盘格参数（在 main() 函数中设置）：
   - chessboard_size: (行数, 列数)，例如 (7, 10) 表示 8×11 棋盘格（内角点）

输出文件（保存到 result 文件夹）：
- corners_left.json：左图角点坐标（像素）
- corners_right.json：右图角点坐标（像素）
- disparities.json：视差值（像素）
- matches_visualization.png：匹配可视化图像

处理流程：
1) 读取左右棋盘格灰度图和掩膜
2) 使用 OpenCV 检测棋盘格角点（仅在掩膜区域内）
3) 亚像素精化角点位置
4) 按行排序角点（从上到下，从左到右）
5) 按索引一一对应匹配角点
6) 计算视差（disparity = x_left - x_right）
7) 保存结果和可视化图像
"""
import os

import cv2
import numpy as np

from .utils.io_utils import read_image_grayscale, find_input_images, find_mask_images, save_results
from .core.corner_detection import find_chessboard_corners, refine_corners
from .core.corner_matching import match_by_relative_coordinates, compute_disparities
from .core.image_processing import crop_image_by_mask
from .utils.visualization import visualize_matches


def main():
    """主函数：执行完整的角点检测、匹配和视差计算流程。"""
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    left_path, right_path = find_input_images(data_dir)
    img_left = read_image_grayscale(left_path)
    img_right = read_image_grayscale(right_path)
    
    print(f"读取图片:\n  左图: {left_path}\n  右图: {right_path}")
    print(f"图像尺寸: 左图 {img_left.shape}, 右图 {img_right.shape}")

    chessboard_size = (12, 16)
    print(f"棋盘格参数: {chessboard_size[0]}×{chessboard_size[1]} (内角点数)")

    print("\n查找棋盘格掩膜...")
    left_mask_path, right_mask_path = find_mask_images(data_dir, left_path, right_path)
    
    if left_mask_path and os.path.exists(left_mask_path):
        print(f"  读取左图掩膜: {left_mask_path}")
        mask_left = read_image_grayscale(left_mask_path)
    else:
        raise FileNotFoundError("未找到左图掩膜文件！")
    
    if right_mask_path and os.path.exists(right_mask_path):
        print(f"  读取右图掩膜: {right_mask_path}")
        mask_right = read_image_grayscale(right_mask_path)
    else:
        raise FileNotFoundError("未找到右图掩膜文件！")
    
    mask_area_left = np.sum(mask_left > 0) / (mask_left.shape[0] * mask_left.shape[1]) * 100
    mask_area_right = np.sum(mask_right > 0) / (mask_right.shape[0] * mask_right.shape[1]) * 100
    print(f"  左图掩膜覆盖: {mask_area_left:.1f}%")
    print(f"  右图掩膜覆盖: {mask_area_right:.1f}%")

    print("\n裁剪左图...")
    img_left_det, mask_left_det, left_offset, left_ratio = crop_image_by_mask(
        img_left, mask_left, padding=50)
    
    # 保存裁剪后的左图掩码
    cropped_left_mask_path = os.path.join(out_dir, "cropped_left_mask.png")
    cv2.imwrite(cropped_left_mask_path, mask_left_det)
    print(f"  保存左图裁剪掩码: {cropped_left_mask_path}")

    print("\n裁剪右图...")
    img_right_det, mask_right_det, right_offset, right_ratio = crop_image_by_mask(
        img_right, mask_right, padding=50)
    
    # 保存裁剪后的右图掩码
    cropped_right_mask_path = os.path.join(out_dir, "cropped_right_mask.png")
    cv2.imwrite(cropped_right_mask_path, mask_right_det)
    print(f"  保存右图裁剪掩码: {cropped_right_mask_path}")

    allow_partial = True
    enhance_contrast = False
    
    print("\n检测左图角点...")
    corners_left = find_chessboard_corners(img_left_det, chessboard_size, 
                                          allow_partial=allow_partial, 
                                          enhance=enhance_contrast,
                                          mask=mask_left_det)
    if corners_left is None:
        raise RuntimeError("左图角点检测失败！")
    if left_offset != (0, 0):
        corners_left[:, 0, 0] += left_offset[0]
        corners_left[:, 0, 1] += left_offset[1]
    
    print("检测右图角点...")
    corners_right = find_chessboard_corners(img_right_det, chessboard_size,
                                            allow_partial=allow_partial,
                                            enhance=enhance_contrast,
                                            mask=mask_right_det)
    if corners_right is None:
        raise RuntimeError("右图角点检测失败！")
    if right_offset != (0, 0):
        corners_right[:, 0, 0] += right_offset[0]
        corners_right[:, 0, 1] += right_offset[1]
    
    print(f"检测到角点: 左图 {len(corners_left)} 个, 右图 {len(corners_right)} 个")

    print("\n亚像素精化角点位置...")
    corners_left = refine_corners(img_left, corners_left)
    corners_right = refine_corners(img_right, corners_right)

    # ========== 角点匹配 ==========
    print("\n进行角点匹配...")
    corners_left_matched, corners_right_matched = match_by_relative_coordinates(
        corners_left, corners_right, chessboard_size
    )
    
    num_matched = len(corners_left_matched)
    print(f"匹配成功: {num_matched} 对")
    
    if num_matched == 0:
        raise RuntimeError("角点匹配未找到任何匹配对！")

    # ========== 计算视差 ==========
    print("\n计算视差...")
    disparities = compute_disparities(corners_left_matched, corners_right_matched)
    
    print(f"视差统计:")
    print(f"  最小值: {disparities.min():.2f} 像素")
    print(f"  最大值: {disparities.max():.2f} 像素")
    print(f"  平均值: {disparities.mean():.2f} 像素")
    print(f"  标准差: {disparities.std():.2f} 像素")

    # ========== 保存结果 ==========
    print("\n保存结果...")
    save_results(out_dir, corners_left_matched, corners_right_matched, disparities)

    # ========== 可视化匹配 ==========
    print("\n生成匹配可视化图...")
    match_vis_path = os.path.join(out_dir, "matches_visualization.png")
    visualize_matches(
        img_left,
        img_right,
        corners_left_matched,
        corners_right_matched,
        disparities,
        match_vis_path
    )
    print(f"  匹配可视化: {match_vis_path}")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
