"""
项目：Stereo_corner_matching
目的：从两张处理后的棋盘格灰度图中检测角点，匹配对应角点，并计算视差。

你需要准备：
1) 把两张处理后的棋盘格灰度图放在 data 文件夹下：
   - 左图：文件名必须包含 “left”（例如 left.png）
   - 右图：文件名必须包含 “right”（例如 right.png）

2) 掩膜文件（必须，放在 data 文件夹下）：
   - 左图掩膜：left_mask.png（或与左图同名的掩膜文件）
   - 右图掩膜：right_mask.png（或与右图同名的掩膜文件）

3) 设置棋盘格参数：
   - 在 main() 函数中修改 chessboard_size 参数
   - 例如：7×9 棋盘格（内角点 6×8 = 48 个）设置为 (6, 8)
   - 例如：9×12 棋盘格（内角点 8×11 = 88 个）设置为 (8, 11)

程序会输出到 result 文件夹：
- corners_left.json：左图角点坐标（像素）
- corners_right.json：右图角点坐标（像素）
        - disparities.json：视差值（像素）
        - matches_visualization.png：左右图拼接的匹配可视化

处理思路：
1) 读取两张处理后的棋盘格灰度图；
2) 读取棋盘格掩膜（必须由 Projector_reflection_diff 生成）；
3) 使用 OpenCV 检测棋盘格角点（findChessboardCorners），只在掩膜区域内检测；
4) 亚像素精化角点位置（cornerSubPix）；
5) 验证角点匹配（极线约束、视差合理性）；
6) 计算视差（disparity = x_left - x_right）；
7) 保存结果和可视化图像。

"""
import os
import json
import glob
from typing import Tuple, Optional

import cv2
import numpy as np
import warnings

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
from io import StringIO

class SuppressOpenCVWarnings:
    """临时抑制 OpenCV 警告的上下文管理器"""
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self._stderr


def read_image_grayscale(path: str) -> np.ndarray:
    """
    读取灰度图片。
    
    从文件路径读取一张灰度图片，支持中文路径和特殊字符。OpenCV 的 imread 函数在遇到中文路径时可能会失败，
    所以先用常规方法读，如果失败了，就用另一种方法（先读成字节再解码）来绕过这个问题。
    
    参数：
        path: 图片文件的完整路径（例如 "data/left.png"）
    
    返回：
        一个 numpy 数组，形状是 (高度, 宽度)，表示灰度像素值。
        每个像素值范围是 0-255（uint8 类型）。
    """
    # 尝试使用 OpenCV 标准方法读取图像（抑制警告信息）
    with SuppressOpenCVWarnings():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # 如果标准方法失败（可能是中文路径问题），使用备用方法
    if img is None:
        try:
            # 先读取为字节流，再解码为图像
            data = np.fromfile(path, dtype=np.uint8)
            with SuppressOpenCVWarnings():
                img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
    
    # 如果两种方法都失败，抛出异常
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    
    return img


def enhance_chessboard_region(img: np.ndarray) -> np.ndarray:
    """
    增强棋盘格区域的对比度，提高角点检测成功率。
    
    如果图像中只有一部分是棋盘格，或者对比度不够，可以通过增强处理提高检测成功率。
    使用自适应直方图均衡化（CLAHE）增强局部对比度。
    
    参数：
        img: 灰度图像（numpy 数组）
    
    返回：
        增强后的灰度图像。
    """
    # 使用 CLAHE（对比度受限的自适应直方图均衡化）增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    return enhanced


def find_chessboard_corners(img: np.ndarray, pattern_size: Tuple[int, int], 
                            allow_partial: bool = True, enhance: bool = True,
                            mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    检测棋盘格角点。
    
    使用 OpenCV 的 findChessboardCorners 函数自动检测棋盘格角点。棋盘格角点是黑白方格的交界处，
    是后续立体匹配和三维重建的关键特征点。
    
    如果图像中只有一部分是棋盘格，可以使用 allow_partial=True 来支持部分棋盘格检测。
    如果图像对比度不够，可以使用 enhance=True 来增强棋盘格区域。
    如果提供了掩膜，只在掩膜标记的区域内检测角点，可以提高检测准确性和速度。
    
    参数：
        img: 灰度图像（numpy 数组）
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
                     例如：7×9 棋盘格（内角点 6×8 = 48 个）设置为 (6, 8)
                     例如：9×12 棋盘格（内角点 8×11 = 88 个）设置为 (8, 11)
        allow_partial: 是否允许部分棋盘格检测，默认 True。如果图像中只有部分棋盘格，设置为 True。
        enhance: 是否增强图像对比度，默认 True。如果图像对比度不够，设置为 True。
        mask: 可选的掩膜（uint8），255 表示搜索区域，0 表示忽略区域。如果提供，只在掩膜区域内检测。
    
    返回：
        如果检测成功，返回角点坐标数组，形状是 (N, 1, 2)，N 是角点数量。
        如果检测失败，返回 None。
    """
    # 预处理：增强对比度（如果需要）
    if enhance:
        img_processed = enhance_chessboard_region(img)
    else:
        img_processed = img.copy()
    
    # 如果提供了掩膜，将掩膜外的区域设为0，限制搜索范围
    if mask is not None:
        # 确保掩膜尺寸与图像匹配
        if mask.shape != img_processed.shape:
            mask = cv2.resize(mask, (img_processed.shape[1], img_processed.shape[0]))
        
        # 将掩膜外的区域设为0（黑色），这样角点检测会忽略这些区域
        img_processed = cv2.bitwise_and(img_processed, mask)
    
    ret = False
    corners: Optional[np.ndarray] = None
    sb_available = hasattr(cv2, "findChessboardCornersSB")
    
    if sb_available:
        try:
            sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
            sb_result = cv2.findChessboardCornersSB(
                img_processed,
                pattern_size,
                flags=sb_flags
            )
            if isinstance(sb_result, tuple):
                sb_ret, sb_corners = sb_result
            else:
                sb_ret, sb_corners = True, sb_result
            if sb_ret and sb_corners is not None:
                sb_corners = np.asarray(sb_corners, dtype=np.float32)
                if sb_corners.ndim == 2:
                    sb_corners = sb_corners.reshape(-1, 1, 2)
                if sb_corners.shape[0] == pattern_size[0] * pattern_size[1]:
                    corners = sb_corners
                    ret = True
                    print("  [提示] 使用 findChessboardCornersSB 成功检测到完整棋盘角点。")
                else:
                    print(f"  [提示] SB 模式检测到 {sb_corners.shape[0]} 个角点，不满足 {pattern_size[0] * pattern_size[1]} 个。")
            else:
                print("  [提示] findChessboardCornersSB 未检测到完整角点。")
        except Exception as sb_err:
            print(f"  [提示] findChessboardCornersSB 调用失败: {sb_err}")
    
    if (not ret):
        # 构建传统算法 flags
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        if allow_partial:
            partial_flag = getattr(cv2, "CALIB_CB_PARTIAL_OK", 0)
            if partial_flag:
                flags += partial_flag
            else:
                print("  [警告] 当前 OpenCV 版本不支持 CALIB_CB_PARTIAL_OK，常规角点检测要求完整棋盘。")
        
        with SuppressOpenCVWarnings():
            ret, corners = cv2.findChessboardCorners(
                img_processed,
                pattern_size,
                flags=flags
            )
        if ret:
            print("  [提示] 常规 findChessboardCorners 检测成功。")
    
    # 如果提供了掩膜，再次验证角点是否在掩膜区域内（双重保险）
    if ret and mask is not None:
        # 提取角点坐标
        corners_pts = corners.reshape(-1, 2).astype(np.int32)
        
        # 检查每个角点是否在掩膜区域内
        valid_mask = np.zeros(len(corners_pts), dtype=bool)
        for i, pt in enumerate(corners_pts):
            y, x = int(pt[1]), int(pt[0])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:  # 在掩膜区域内
                    valid_mask[i] = True
        
        # 如果有效角点太少，返回 None
        if valid_mask.sum() < pattern_size[0] * pattern_size[1] * 0.3:
            return None
        
        # 过滤角点
        corners = corners[valid_mask]
    
    if not ret:
        return None
    
    return corners


def refine_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    亚像素精化角点位置。
    
    使用 cornerSubPix 函数将角点位置从整数像素精化到亚像素级（0.1像素精度），
    基于局部灰度梯度优化，提高角点定位精度。
    
    参数：
        img: 灰度图像（numpy 数组）
        corners: 初始角点坐标，形状是 (N, 1, 2)
    
    返回：
        精化后的角点坐标，形状与输入相同。
    """
    # 设置亚像素精化的搜索窗口和终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    winSize = (11, 11)  # 搜索窗口大小
    zeroZone = (-1, -1)  # 死区大小（不使用）
    
    # 执行亚像素精化
    refined = cv2.cornerSubPix(img, corners, winSize, zeroZone, criteria)
    
    return refined


def validate_correspondence(corners_left: np.ndarray, corners_right: np.ndarray, 
                            epipolar_thresh: float = 1.0, 
                            disparity_range: Tuple[float, float] = (0, 500)) -> Tuple[np.ndarray, np.ndarray]:
    """
    验证角点对应关系。
    
    对于棋盘格，角点自动对应（左图第i个角点对应右图第i个角点），但需要验证对应关系的合理性。
    使用极线约束（对应点的纵坐标差应很小）和视差合理性（视差应在预设范围内）进行验证。
    
    参数：
        corners_left: 左图角点坐标，形状是 (N, 1, 2)
        corners_right: 右图角点坐标，形状是 (N, 1, 2)
        epipolar_thresh: 极线约束阈值（像素），默认 1.0。对应点的纵坐标差应小于此值。
        disparity_range: 视差范围 (min, max)，默认 (0, 500)。视差应在此范围内。
    
    返回：
        (valid_left, valid_right) 两个角点数组，只包含通过验证的角点对。
    """
    # 提取坐标（去除多余的维度）
    left_pts = corners_left.reshape(-1, 2)  # (N, 2)
    right_pts = corners_right.reshape(-1, 2)  # (N, 2)
    
    # 计算视差（x_left - x_right）
    disparities = left_pts[:, 0] - right_pts[:, 0]
    
    # 计算纵坐标差（用于极线约束验证）。由于相机未做极线校正，先减去中值偏移以消除整体错层。
    y_offset = np.median(left_pts[:, 1] - right_pts[:, 1])
    y_diff = np.abs((left_pts[:, 1] - right_pts[:, 1]) - y_offset)
    
    # 打印统计信息，方便调参
    print(f"  极线差统计: min={y_diff.min():.2f}, max={y_diff.max():.2f}, mean={y_diff.mean():.2f}, 阈值={epipolar_thresh}, 纵向修正={y_offset:.2f}")
    print(f"  视差统计: min={disparities.min():.2f}, max={disparities.max():.2f}, mean={disparities.mean():.2f}, 范围={disparity_range}")
    
    # 验证条件
    valid_mask = (
        (y_diff < epipolar_thresh) &  # 极线约束：纵坐标差应很小
        (disparities >= disparity_range[0]) &  # 视差最小值
        (disparities <= disparity_range[1])  # 视差最大值
    )
    
    # 提取有效角点
    valid_left = corners_left[valid_mask]
    valid_right = corners_right[valid_mask]
    
    return valid_left, valid_right


def compute_disparities(corners_left: np.ndarray, corners_right: np.ndarray) -> np.ndarray:
    """
    计算视差值。
    
    对于每对匹配的角点，计算视差 = x_left - x_right。视差是立体视觉中计算深度的关键参数。
    
    参数：
        corners_left: 左图角点坐标，形状是 (N, 1, 2)
        corners_right: 右图角点坐标，形状是 (N, 1, 2)
    
    返回：
        视差值数组，形状是 (N,)，单位是像素。
    """
    # 提取 x 坐标
    x_left = corners_left[:, 0, 0]  # (N,)
    x_right = corners_right[:, 0, 0]  # (N,)
    
    # 计算视差
    disparities = x_left - x_right
    
    return disparities


def visualize_matches(img_left: np.ndarray,
                      img_right: np.ndarray,
                      corners_left: np.ndarray,
                      corners_right: np.ndarray,
                      disparities: np.ndarray,
                      out_path: str,
                      max_pair_labels: int = 60) -> None:
    """
    生成匹配角点与视差的可视化图。

    将左右图横向拼接，在图上绘制匹配角点、相同编号及视差，方便人工核查。

    参数：
        img_left: 左图（灰度或彩色）
        img_right: 右图（灰度或彩色）
        corners_left: 左图有效角点 (N, 1, 2)
        corners_right: 右图有效角点 (N, 1, 2)
        disparities: 对应的视差数组 (N,)
        out_path: 输出图片路径
        max_pair_labels: 图像上最多标注的编号/视差对，避免遮挡
    """
    if corners_left.size == 0 or corners_right.size == 0 or disparities.size == 0:
        print("  [提示] 没有可视化的匹配角点，跳过可视化。")
        return

    if img_left.ndim == 2:
        left_vis = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = img_left.copy()

    if img_right.ndim == 2:
        right_vis = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = img_right.copy()

    h = max(left_vis.shape[0], right_vis.shape[0])
    w = left_vis.shape[1] + right_vis.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:left_vis.shape[0], :left_vis.shape[1]] = left_vis
    canvas[:right_vis.shape[0], left_vis.shape[1]:left_vis.shape[1] + right_vis.shape[1]] = right_vis

    offset_x = left_vis.shape[1]
    disp_min = float(np.min(disparities))
    disp_max = float(np.max(disparities))
    disp_span = max(disp_max - disp_min, 1e-6)

    corners_left_flat = corners_left.reshape(-1, 2)
    corners_right_flat = corners_right.reshape(-1, 2)

    for idx, (pt_l, pt_r, disp) in enumerate(zip(corners_left_flat, corners_right_flat, disparities)):
        alpha = (float(disp) - disp_min) / disp_span
        color = (
            int(255 * (1.0 - alpha)),      # Blue channel decreases with disparity
            int(128),                      # Constant green for visibility
            int(255 * alpha)               # Red channel increases with disparity
        )

        pt_l_int = tuple(np.round(pt_l).astype(int))
        pt_r_int = tuple(np.round(pt_r).astype(int))
        pt_r_shifted = (pt_r_int[0] + offset_x, pt_r_int[1])

        cv2.circle(canvas, pt_l_int, 5, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_r_shifted, 5, color, 2, lineType=cv2.LINE_AA)

        if idx < max_pair_labels:
            pair_id = str(idx + 1)
            cv2.putText(canvas, pair_id, (pt_l_int[0] + 6, pt_l_int[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, pair_id, (pt_r_shifted[0] + 6, pt_r_shifted[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def find_input_images(data_dir: str) -> Tuple[str, str]:
    """
    在 data 目录里查找文件名包含 “left” 和 “right” 的棋盘格图片。
    
    如果缺少任意一侧，直接报错，不再自动回退到其它文件名。
    """
    # 支持的图片格式
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    
    # 扫描目录，收集所有图片文件
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, e)))
    
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到图片")
    
    # 优先匹配：查找文件名中包含 "left" 和 "right" 的图片
    left_candidates = sorted(
        [f for f in files if "left" in os.path.basename(f).lower()])
    right_candidates = sorted(
        [f for f in files if "right" in os.path.basename(f).lower()])
    
    if not left_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'left' 的图片")
    if not right_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'right' 的图片")
    
    return left_candidates[0], right_candidates[0]


def find_mask_images(data_dir: str, left_path: str, right_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    在 data 目录中查找掩膜文件。
    
    查找顺序：
        1. 固定命名：left_mask.* / right_mask.*
        2. 与图像同名：<原图文件名>_mask.*
    
    如果缺少任意一侧的掩膜，返回 None，交由调用方报错。
    """
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    
    def _find_mask(img_path: str, default_prefix: str) -> Optional[str]:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = []
        for ext in exts:
            candidates.append(os.path.join(data_dir, f"{default_prefix}{ext}"))
            candidates.append(os.path.join(data_dir, f"{base_name}_mask{ext}"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None
    
    left_mask = _find_mask(left_path, "left_mask")
    right_mask = _find_mask(right_path, "right_mask")
    
    return left_mask, right_mask


def crop_image_by_mask(img: np.ndarray,
                       mask: np.ndarray,
                       min_ratio: float = 0.12,
                       padding: int = 40,
                       focus_largest_component: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    """
    当掩膜覆盖比例很小时，自动裁剪到掩膜外接矩形，并保留一定 padding。
    默认会先保留掩膜中面积最大的连通域，避免零散噪声把外接矩形拉得过大。

    返回裁剪后的图像、掩膜、左上角偏移量 (x_offset, y_offset) 以及原始覆盖率。
    如果覆盖率已足够大，则返回原图。
    """
    if mask is None or img is None:
        return img, mask, (0, 0), 0.0

    mask_binary = (mask > 0).astype(np.uint8)

    if focus_largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels > 1:
            largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask_binary = (labels == largest_idx).astype(np.uint8)

    ratio = float(np.sum(mask_binary)) / mask_binary.size
    if ratio == 0:
        return img, mask, (0, 0), 0.0

    mask_filtered = (mask_binary * 255).astype(np.uint8)
    if ratio >= min_ratio:
        return img, mask_filtered, (0, 0), ratio

    ys, xs = np.where(mask_binary > 0)
    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding, img.shape[0])
    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, img.shape[1])

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask_filtered[y1:y2, x1:x2]

    return cropped_img, cropped_mask, (x1, y1), ratio


def save_results(out_dir: str, corners_left: np.ndarray, corners_right: np.ndarray, 
                 disparities: np.ndarray) -> None:
    """
    保存角点坐标和视差值到 JSON 文件。
    
    将检测到的角点坐标和计算出的视差值保存为 JSON 格式，便于后续处理和三维重建。
    
    参数：
        out_dir: 输出目录路径
        corners_left: 左图角点坐标，形状是 (N, 1, 2)
        corners_right: 右图角点坐标，形状是 (N, 1, 2)
        disparities: 视差值数组，形状是 (N,)
    
    返回：
        无
    """
    # 转换角点坐标为列表格式
    left_pts = corners_left.reshape(-1, 2).tolist()
    right_pts = corners_right.reshape(-1, 2).tolist()
    disparities_list = disparities.tolist()
    
    # 保存左图角点
    left_path = os.path.join(out_dir, "corners_left.json")
    with open(left_path, 'w', encoding='utf-8') as f:
        json.dump(left_pts, f, indent=2, ensure_ascii=False)
    
    # 保存右图角点
    right_path = os.path.join(out_dir, "corners_right.json")
    with open(right_path, 'w', encoding='utf-8') as f:
        json.dump(right_pts, f, indent=2, ensure_ascii=False)
    
    # 保存视差值
    disp_path = os.path.join(out_dir, "disparities.json")
    with open(disp_path, 'w', encoding='utf-8') as f:
        json.dump(disparities_list, f, indent=2, ensure_ascii=False)
    
    print(f"保存结果:\n  左图角点: {left_path}\n  右图角点: {right_path}\n  视差值: {disp_path}")


def main():
    """
    主函数：执行完整的角点检测、匹配和视差计算流程。
    
    这是程序的入口，负责协调所有步骤，从读取图片到输出结果。通过一系列图像处理步骤，
    从两张处理后的棋盘格灰度图中检测角点，匹配对应角点，并计算视差，为后续的三维重建做准备。
    
    原理：
        1. 设置路径：找到 data 和 result 目录
        2. 读取图片：自动找到左图和右图
        3. 读取掩膜：必须使用 Projector_reflection_diff 生成的掩膜文件，如果找不到则报错
        4. 检测角点：使用 OpenCV 检测棋盘格角点，只在掩膜区域内检测
        5. 亚像素精化：提高角点定位精度
        6. 验证匹配：使用极线约束和视差合理性验证角点对应关系
        7. 计算视差：disparity = x_left - x_right
        8. 保存结果：角点坐标和视差值保存为 JSON 文件
        9. 可视化：生成并保存可视化图像
    
    参数：
        无（所有参数通过函数调用时的默认值设置）
    
    返回：
        无（结果保存到 result 目录）
    
    输出说明：
        - corners_left.json：左图角点坐标（像素），格式为 [[x1, y1], [x2, y2], ...]
        - corners_right.json：右图角点坐标（像素），格式同上
        - disparities.json：视差值（像素），格式为 [d1, d2, ...]
    
    注意事项：
        - 如果 data 目录里没有图片或只有一张，程序会报错
        - 必须提供由 Projector_reflection_diff 生成的掩膜文件，如果找不到则报错
        - 如果角点检测失败，程序会报错并提示检查棋盘格参数
        - 需要根据实际棋盘格尺寸修改 chessboard_size 参数
        - 如果图像中只有部分棋盘格，程序默认支持部分检测（allow_partial=True）
        - 如果图像对比度不够，程序默认会增强对比度（enhance_contrast=True）
        - 掩膜用于限制角点检测的搜索范围，提高准确性和速度
    """
    # ========== 初始化：设置输入输出目录 ==========
    proj_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_root, "data")
    out_dir = os.path.join(proj_root, "result")
    os.makedirs(out_dir, exist_ok=True)

    # ========== 读取输入图片 ==========
    # 自动识别 data 目录中的左图和右图
    left_path, right_path = find_input_images(data_dir)
    
    # 读取图片（支持中文路径）
    img_left = read_image_grayscale(left_path)
    img_right = read_image_grayscale(right_path)
    
    print(f"读取图片:\n  左图: {left_path}\n  右图: {right_path}")
    print(f"图像尺寸: 左图 {img_left.shape}, 右图 {img_right.shape}")

    # ========== 设置棋盘格参数 ==========
    # 根据实际棋盘格尺寸修改此参数
    # 例如：7×9 棋盘格（内角点 6×8 = 48 个）设置为 (6, 8)
    # 例如：9×12 棋盘格（内角点 8×11 = 88 个）设置为 (8, 11)
    chessboard_size = (7, 10)  
    
    print(f"棋盘格参数: {chessboard_size[0]}×{chessboard_size[1]} (内角点数)")

    # ========== 步骤0：读取棋盘格掩膜 ==========
    # 必须使用 Projector_reflection_diff 生成的掩膜，限制角点检测的搜索范围
    print("\n查找棋盘格掩膜...")
    # 尝试找到掩膜文件
    left_mask_path, right_mask_path = find_mask_images(data_dir, left_path, right_path)
    
    # 读取左图掩膜
    if left_mask_path and os.path.exists(left_mask_path):
        print(f"  读取左图掩膜: {left_mask_path}")
        mask_left = read_image_grayscale(left_mask_path)
    else:
        raise FileNotFoundError(
            "未找到左图掩膜文件！请在 data 目录提供 left_mask.(png/jpg) "
            "或与左图同名的 *_mask.(png/jpg) 文件。")
    
    # 读取右图掩膜
    if right_mask_path and os.path.exists(right_mask_path):
        print(f"  读取右图掩膜: {right_mask_path}")
        mask_right = read_image_grayscale(right_mask_path)
    else:
        raise FileNotFoundError(
            "未找到右图掩膜文件！请在 data 目录提供 right_mask.(png/jpg) "
            "或与右图同名的 *_mask.(png/jpg) 文件。")
    
    # 统计掩膜覆盖的区域，并在覆盖过小时裁剪 ROI
    mask_area_left = np.sum(mask_left > 0) / (mask_left.shape[0] * mask_left.shape[1]) * 100
    mask_area_right = np.sum(mask_right > 0) / (mask_right.shape[0] * mask_right.shape[1]) * 100
    print(f"  左图掩膜覆盖: {mask_area_left:.1f}%")
    print(f"  右图掩膜覆盖: {mask_area_right:.1f}%")

    img_left_det, mask_left_det, left_offset, left_ratio = crop_image_by_mask(
        img_left, mask_left, min_ratio=0.12, padding=50)
    if left_offset != (0, 0):
        x1, y1 = left_offset
        x2 = x1 + img_left_det.shape[1]
        y2 = y1 + img_left_det.shape[0]
        print(f"  左图掩膜覆盖较小（{left_ratio*100:.1f}%），裁剪 ROI 到 x[{x1}:{x2}], y[{y1}:{y2}]")
    else:
        print("  左图掩膜覆盖充分，使用整幅图像")

    img_right_det, mask_right_det, right_offset, right_ratio = crop_image_by_mask(
        img_right, mask_right, min_ratio=0.12, padding=50)
    if right_offset != (0, 0):
        x1, y1 = right_offset
        x2 = x1 + img_right_det.shape[1]
        y2 = y1 + img_right_det.shape[0]
        print(f"  右图掩膜覆盖较小（{right_ratio*100:.1f}%），裁剪 ROI 到 x[{x1}:{x2}], y[{y1}:{y2}]")
    else:
        print("  右图掩膜覆盖充分，使用整幅图像")

    # ========== 步骤1：检测角点 ==========
    # 角点检测与匹配参数
    allow_partial = True   # 允许部分棋盘格检测（如果图像中只有部分棋盘格，设置为 True）
    enhance_contrast = True  # 增强对比度（如果图像对比度不够，设置为 True）
    epipolar_thresh = 8.0    # 左右图像未严格极线校正，放宽到 8 像素
    disparity_range = (-200, 800)  # 根据基线和拍摄距离放宽视差范围
    
    print("\n检测左图角点...")
    print(f"  允许部分棋盘格: {allow_partial}, 增强对比度: {enhance_contrast}")
    corners_left = find_chessboard_corners(img_left_det, chessboard_size, 
                                          allow_partial=allow_partial, 
                                          enhance=enhance_contrast,
                                          mask=mask_left_det)
    if corners_left is None:
        raise RuntimeError(f"左图角点检测失败！请检查：\n"
                          f"  1. 棋盘格参数是否正确（当前: {chessboard_size}）\n"
                          f"  2. 图像中是否包含棋盘格（完整或部分）\n"
                          f"  3. 图像对比度是否足够\n"
                          f"  4. 如果图像中只有部分棋盘格，确保 allow_partial=True\n"
                          f"  5. 如果对比度不够，确保 enhance_contrast=True")
    if left_offset != (0, 0):
        corners_left[:, 0, 0] += left_offset[0]
        corners_left[:, 0, 1] += left_offset[1]
    
    print("检测右图角点...")
    corners_right = find_chessboard_corners(img_right_det, chessboard_size,
                                            allow_partial=allow_partial,
                                            enhance=enhance_contrast,
                                            mask=mask_right_det)
    if corners_right is None:
        raise RuntimeError(f"右图角点检测失败！请检查：\n"
                          f"  1. 棋盘格参数是否正确（当前: {chessboard_size}）\n"
                          f"  2. 图像中是否包含棋盘格（完整或部分）\n"
                          f"  3. 图像对比度是否足够\n"
                          f"  4. 如果图像中只有部分棋盘格，确保 allow_partial=True\n"
                          f"  5. 如果对比度不够，确保 enhance_contrast=True")
    if right_offset != (0, 0):
        corners_right[:, 0, 0] += right_offset[0]
        corners_right[:, 0, 1] += right_offset[1]
    
    print(f"检测到角点: 左图 {len(corners_left)} 个, 右图 {len(corners_right)} 个")
    
    # 检查检测到的角点数量是否合理
    expected_count = chessboard_size[0] * chessboard_size[1]
    if len(corners_left) < expected_count * 0.5:
        print(f"  警告: 左图只检测到 {len(corners_left)}/{expected_count} 个角点，可能只有部分棋盘格")
    if len(corners_right) < expected_count * 0.5:
        print(f"  警告: 右图只检测到 {len(corners_right)}/{expected_count} 个角点，可能只有部分棋盘格")

    # ========== 步骤2：亚像素精化 ==========
    print("\n亚像素精化角点位置...")
    corners_left = refine_corners(img_left, corners_left)
    corners_right = refine_corners(img_right, corners_right)

    # ========== 步骤3：验证匹配 ==========
    print("\n验证角点对应关系...")
    corners_left_valid, corners_right_valid = validate_correspondence(
        corners_left, corners_right,
        epipolar_thresh=epipolar_thresh,
        disparity_range=disparity_range
    )
    
    num_valid = len(corners_left_valid)
    num_total = len(corners_left)
    print(f"有效角点对: {num_valid}/{num_total} ({100*num_valid/num_total:.1f}%)")
    
    if num_valid == 0:
        raise RuntimeError("没有有效的角点对！请检查：\n"
                          f"  1. 左右图像是否来自同一场景\n"
                          f"  2. 极线约束阈值是否合适（当前: "
                          f"{epipolar_thresh} 像素）\n"
                          f"  3. 视差范围是否合理（当前: {disparity_range} 像素）")

    # ========== 步骤4：计算视差 ==========
    print("\n计算视差...")
    disparities = compute_disparities(corners_left_valid, corners_right_valid)
    
    print(f"视差统计:")
    print(f"  最小值: {disparities.min():.2f} 像素")
    print(f"  最大值: {disparities.max():.2f} 像素")
    print(f"  平均值: {disparities.mean():.2f} 像素")
    print(f"  标准差: {disparities.std():.2f} 像素")

    # ========== 步骤5：保存结果 ==========
    print("\n保存结果...")
    save_results(out_dir, corners_left_valid, corners_right_valid, disparities)

    # ========== 步骤6：可视化匹配 ==========
    print("\n生成匹配可视化图...")
    match_vis_path = os.path.join(out_dir, "matches_visualization.png")
    visualize_matches(
        img_left,
        img_right,
        corners_left_valid,
        corners_right_valid,
        disparities,
        match_vis_path
    )
    print(f"  匹配可视化: {match_vis_path}")
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

