"""
投影反射差分主程序

从两张同一位置拍摄的图片中，提取"投影灯打开时"才出现的那部分反射图案，
让棋盘格等投影图案在图片里更清楚，便于后续角点检测与三维重建。

输入要求：
1) 图片文件（放在 data 文件夹下）：
   - 关灯图（只含环境反射）：env.jpg
   - 开灯图（环境反射 + 投影反射）：mix.jpg

输出文件（保存到 result 文件夹）：
- result_detect.png：给算法使用的"更清晰"的灰度图（已经做了对齐、亮度匹配、相减与背景抑制）
- result_mask.png：棋盘格区域的掩膜图（用于后续角点检测时限制搜索范围）

处理流程：
1) 把关灯图的位置对齐到开灯图（避免相机轻微移动引起的错位）
2) 调整两张图的整体亮度/颜色，让它们可比（避免相机自动曝光的影响）
3) 用"开灯图 - 关灯图"的思路，算出只有投影才带来的那部分反射
4) 只在棋盘格所在区域里，去掉缓慢变化的背景，只保留细节纹理，让角点更明显
5) 保存一张"检测用"的灰度图
"""
import os
import cv2
import numpy as np

from .utils.io_utils import read_image_bgr, safe_imwrite, find_input_images
from .core.alignment import resize_to_match, ecc_register
from .core.brightness_matching import build_fit_mask, channelwise_compensated_diff
from .core.chessboard_processing import (
    build_chess_mask_from_proj,
    suppress_background_in_mask,
    enhance_chessboard_contrast
)


def process_projection_diff(env_path: str, mix_path: str, output_dir: str) -> tuple:
    """
    处理投影反射差分，生成棋盘格图和掩码图
    
    参数:
        env_path: 环境图路径
        mix_path: 混合图路径
        output_dir: 输出目录
    
    返回:
        (chessboard_img, mask_img) 棋盘格图和掩码图
    """
    # 读取图片
    img_env = read_image_bgr(env_path)
    img_mix = read_image_bgr(mix_path)
    
    # 尺寸对齐
    img_env = resize_to_match(img_env, (img_mix.shape[0], img_mix.shape[1]))
    
    # 几何对齐（ECC配准）
    img_env_aligned = ecc_register(img_env, img_mix, motion_model="homography")
    
    # 构建拟合掩膜
    mask_fit = build_fit_mask(img_env_aligned, img_mix, diff_thresh=12)
    
    # 提取纯投影反射
    r_proj_bgr = channelwise_compensated_diff(img_env_aligned, img_mix, mask_fit)
    r_proj_gray = cv2.cvtColor(r_proj_bgr, cv2.COLOR_BGR2GRAY)
    
    # 构建棋盘格掩膜
    chess_mask = build_chess_mask_from_proj(r_proj_gray, block_size=31, C=-5, min_area=200)
    
    # 背景抑制
    r_proj_hf = suppress_background_in_mask(r_proj_gray, chess_mask, ksize=51)
    
    # 对比度增强
    chessboard_img = enhance_chessboard_contrast(r_proj_hf, chess_mask, clip_limit=2.0, tile_grid_size=(8, 8))
    
    return chessboard_img, chess_mask


def main():
    """
    主函数：执行完整的投影反射差分流程。
    
    这是程序的入口，负责协调所有步骤，从读取图片到输出结果。通过一系列图像处理步骤，
    从两张图片（关灯图和开灯图）中提取出只有投影才带来的反射图案，并增强棋盘格区域的对比度，
    便于后续的角点检测和三维重建。
    
    原理：
        1. 设置路径：找到 data 和 result 目录
        2. 读取图片：自动找到关灯图和开灯图
        3. 尺寸对齐：让两张图大小一致
        4. 几何对齐：把关灯图对齐到开灯图（ECC 配准）
        5. 亮度匹配：找出稳定区域，计算 beta、gamma 参数
        6. 提取纯投影反射：做补偿相减，得到只有投影才带来的反射
        7. 转灰度：转为灰度图便于后续处理
        8. 找出棋盘格区域：自动生成掩膜
        9. 背景抑制：在棋盘格区域内去掉缓慢变化的背景
        10. 对比度增强：在棋盘格区域内使用 CLAHE 增强对比度，使角点更突出
        11. 保存结果：result_detect.png（用于角点检测的灰度图）
    
    参数：
        无（所有参数通过函数调用时的默认值设置）
    
    返回：
        无（结果保存到 result 目录）
    
    输出说明：
        - result_detect.png：这是给算法用的，已经做了对齐、亮度匹配、相减、背景抑制和对比度增强，
                            棋盘格的角点会更清晰，适合后续的角点检测和三维重建。
        - result_mask.png：自动生成的棋盘格掩膜，Stereo_corner_matching 会直接引用它来限制角点搜索区域。
    
    注意事项：
        - 如果 data 目录里没有图片或只有一张，程序会报错
        - 如果对齐失败（比如两张图差异太大），程序会继续运行，但结果可能不理想
        - 所有参数都使用默认值，适合大多数情况。如果需要调整，可以修改函数调用时的参数
    """
    # ========== 初始化：设置输入输出目录 ==========
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)
    print("=== 投影反射差分流程启动 ===")
    print(f"工作目录: {proj_root}")
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {out_dir}")

    # ========== 读取输入图片 ==========
    # 自动识别 data 目录中的关灯图和开灯图
    env_path, mix_path = find_input_images(data_dir)
    print("\n[步骤0] 读取输入图片")
    print(f"  关灯图: {env_path}")
    print(f"  开灯图: {mix_path}")
    
    # 读取图片（支持中文路径）
    img_env = read_image_bgr(env_path)
    img_mix = read_image_bgr(mix_path)
    print(f"  原始尺寸: env={img_env.shape}, mix={img_mix.shape}")

    # ========== 步骤1：尺寸对齐 ==========
    # 将关灯图调整到与开灯图相同的尺寸
    print("\n[步骤1] 尺寸对齐 -> resize_to_match")
    img_env = resize_to_match(img_env, (img_mix.shape[0], img_mix.shape[1]))
    print(f"  调整后尺寸: env={img_env.shape}, mix={img_mix.shape}")
    
    # ========== 步骤2：几何对齐 ==========
    # 使用 ECC 算法将关灯图对齐到开灯图的坐标系，消除相机移动造成的错位
    print("\n[步骤2] 几何对齐 -> ECC 配准 (motion=homography)")
    img_env_aligned = ecc_register(img_env, img_mix, motion_model="homography")
    print("  ECC 配准完成")

    # ========== 步骤3：构建拟合掩膜 ==========
    # 找出"变化很小"的稳定区域，用于后续的亮度匹配（排除棋盘格区域）
    print("\n[步骤3] 构建拟合掩膜 -> build_fit_mask")
    mask_fit = build_fit_mask(img_env_aligned, img_mix, diff_thresh=12)
    fit_ratio = np.sum(mask_fit > 0) / mask_fit.size * 100
    print(f"  掩膜覆盖率: {fit_ratio:.1f}% (diff_thresh=12)")
    
    # ========== 步骤4：提取纯投影反射 ==========
    # 对每个颜色通道进行亮度匹配后相减，得到纯投影反射的彩色图
    print("\n[步骤4] 提取纯投影反射 -> channelwise_compensated_diff")
    r_proj_bgr = channelwise_compensated_diff(img_env_aligned, img_mix, mask_fit)
    print("  亮度匹配+相减完成")
    
    # 转换为灰度图，便于后续处理
    r_proj_gray = cv2.cvtColor(r_proj_bgr, cv2.COLOR_BGR2GRAY)

    # ========== 步骤5：构建棋盘格掩膜 ==========
    # 自动识别棋盘格所在的大致区域
    print("\n[步骤5] 构建棋盘格掩膜 -> build_chess_mask_from_proj")
    chess_mask = build_chess_mask_from_proj(r_proj_gray, block_size=31, C=-5, min_area=200)
    mask_ratio = np.sum(chess_mask > 0) / chess_mask.size * 100
    print(f"  棋盘格掩膜覆盖率: {mask_ratio:.1f}% (block=31, C=-5, min_area=200)")
    
    # ========== 步骤6：背景抑制 ==========
    # 在棋盘格区域内使用高斯高通滤波，去除缓慢变化的背景，只保留细节纹理
    print("\n[步骤6] 背景抑制 -> suppress_background_in_mask")
    r_proj_hf = suppress_background_in_mask(r_proj_gray, chess_mask, ksize=51)
    print("  背景抑制完成 (ksize=51)")
    
    # ========== 步骤7：对比度增强 ==========
    # 在棋盘格区域内使用 CLAHE 增强对比度，使角点更突出
    # 使用较温和的参数，避免过度增强导致角点检测失败
    print("\n[步骤7] 对比度增强 -> enhance_chessboard_contrast")
    r_proj_enhanced = enhance_chessboard_contrast(r_proj_hf, chess_mask, clip_limit=2.0, tile_grid_size=(8, 8))
    print("  对比度增强完成 (clip_limit=2.0, tile_grid=(8,8))")

    # ========== 保存输出结果 ==========
    # 保存检测用灰度图：用于角点检测和三维重建（已做背景抑制和对比度增强）
    detect_out = os.path.join(out_dir, "result_detect.png")
    safe_imwrite(detect_out, r_proj_enhanced)
    
    # 保存棋盘格掩膜：用于后续角点检测时限制搜索范围
    mask_out = os.path.join(out_dir, "result_mask.png")
    safe_imwrite(mask_out, chess_mask)
    
    print("\n[完成] 文件已保存：")
    print(f"  棋盘格灰度: {detect_out}")
    print(f"  棋盘格掩膜: {mask_out}")
    print("=== 处理结束 ===")


if __name__ == "__main__":
    main()
