"""
项目：Projector_reflection_diff
目的：从两张同一位置拍摄的图片中，提取“投影灯打开时”才出现的那部分反射图案，
     让棋盘格等投影图案在图片里更清楚，便于后续角点检测与三维重建。

你需要准备：
1) 把两张图片放在 data 文件夹下：
   - 关灯图（只含环境反射）：env.jpg
   - 开灯图（环境反射 + 投影反射）：mix.jpg
   如果文件名里没有 env/mix，程序会按文件名排序取前两张。

程序会输出到 result 文件夹：
- result_detect.png：给算法使用的“更清晰”的灰度图（已经做了对齐、亮度匹配、相减与背景抑制）
- result_mask.png：棋盘格区域的掩膜图（用于后续角点检测时限制搜索范围）

处理思路：
1) 把关灯图的位置对齐到开灯图（避免相机轻微移动引起的错位）；
2) 调整两张图的整体亮度/颜色，让它们可比（避免相机自动曝光的影响）；
3) 用“开灯图 - 关灯图”的思路，算出只有投影才带来的那部分反射；
4) 只在棋盘格所在区域里，去掉缓慢变化的背景，只保留细节纹理，让角点更明显；
5) 保存一张“检测用”的灰度图。

"""
import os
import glob
from typing import Tuple
import warnings

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import numpy as np

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


def read_image_bgr(path: str) -> np.ndarray:
    """
    读取彩色图片（BGR 顺序）。
    
    从文件路径读取一张彩色图片，支持中文路径和特殊字符。OpenCV 的 imread 函数在遇到中文路径时可能会失败，
    所以先用常规方法读，如果失败了，就用另一种方法（先读成字节再解码）来绕过这个问题。
    
    参数：
        path: 图片文件的完整路径（例如 "data/env.jpg"）
    
    返回：
        一个 numpy 数组，形状是 (高度, 宽度, 3)，表示 BGR 三个颜色通道的像素值。
        每个像素值范围是 0-255（uint8 类型）。
    """
    # 尝试使用 OpenCV 标准方法读取图像（抑制警告信息）
    with SuppressOpenCVWarnings():
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    # 如果标准方法失败（可能是中文路径问题），使用备用方法
    if img is None:
        try:
            # 先读取为字节流，再解码为图像
            data = np.fromfile(path, dtype=np.uint8)
            with SuppressOpenCVWarnings():
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    
    # 如果两种方法都失败，抛出异常
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    
    return img


def safe_imwrite(path: str, image: np.ndarray) -> None:
    """
    可靠保存图片到文件，支持中文路径和特殊字符。
    
    把处理好的图片数组保存成文件，即使路径里有中文或特殊字符也能正常保存。
    和读取一样，OpenCV 的 imwrite 在遇到中文路径时可能失败，这里先用 imencode（编码成字节）再写到文件，
    如果失败了再回退到常规方法。
    
    参数：
        path: 要保存的文件路径（例如 "result/result_detect.png"）
        image: 要保存的图片数组（numpy 数组，通常是 uint8 类型）
    
    返回：
        无
    """
    # 提取文件扩展名，用于确定编码格式
    ext = os.path.splitext(path)[1].lower() or ".png"
    
    # 优先使用 imencode + tofile 方法（支持中文路径）
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        # 将编码后的缓冲区直接写入文件
        buf.tofile(path)
        return
    except Exception:
        pass
    
    # 备用方法：使用标准 imwrite（可能不支持中文路径）
    if not cv2.imwrite(path, image):
        raise IOError(f"写入图像失败: {path}")


def resize_to_match(src: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    调整图片尺寸，让它和另一张图片一样大。
    
    如果两张图片的宽度或高度不一样，先把它们调整成一样大小，才能做后续的对齐和相减操作。
    关灯图和开灯图可能用不同分辨率拍摄，或者相机设置不同，在做像素级操作（比如相减）之前，必须先把尺寸统一。

    参数：
        src: 源图片（numpy 数组）
        target_shape: 目标尺寸，格式是 (高度, 宽度)
    
    返回：
        调整后的图片。如果尺寸已经一样，就直接返回原图（不浪费计算）。
    """
    h, w = target_shape
    
    # 如果尺寸已匹配，直接返回原图
    if src.shape[0] == h and src.shape[1] == w:
        return src
    
    # 使用线性插值调整尺寸
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def ecc_register(moving: np.ndarray, fixed: np.ndarray, motion_model: str = "homography") -> np.ndarray:
    """
    把关灯图的位置“对齐”到开灯图，消除相机轻微移动造成的错位。
    
    即使拍摄时相机有轻微移动或角度变化，也能让两张图片的像素点“对得上”，这样后续相减时不会出现重影或模糊。
    无人机或手持拍摄时，很难保证两张图片完全对齐。如果直接相减，错位的地方会出现“重影”，影响结果质量。
    对齐后，同一位置的像素才真正对应同一块玻璃区域。
    
    原理：
        1. 将两张图片转换为灰度图并归一化到 [0, 1] 范围（ECC 算法要求）
        2. 根据运动模型选择变换类型（单应性或仿射）和初始变换矩阵
        3. 使用 ECC（增强相关系数）算法，通过最大化两张图的相似度来找到最佳变换矩阵
        4. 迭代优化变换参数，直到对齐精度达到要求（最大迭代次数 200，精度阈值 1e-6）
        5. 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
        6. 如果 ECC 失败（比如两张图差异太大），返回原图不做处理
    
    参数：
        moving: 需要被对齐的图片（关灯图）
        fixed:  参考图片（开灯图），moving 会被对齐到这个图片的坐标系
        motion_model: 对齐模型，默认 "homography"（单应性，能处理轻微视角变化）
                      也可以选 "affine"（仿射，只能处理平移、旋转、缩放）
    
    返回：
        对齐后的关灯图，尺寸和 fixed 一样。
    """
    # 转换为灰度图并归一化到 [0, 1] 范围（ECC 算法要求）
    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 根据运动模型选择变换类型和初始变换矩阵
    if motion_model == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)  # 3x3 单应矩阵
    else:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)  # 2x3 仿射矩阵

    # 设置迭代终止条件：最大迭代次数 200，精度阈值 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    
    # 使用 ECC 算法计算最优变换矩阵
    try:
        _, warp_matrix = cv2.findTransformECC(
            templateImage=fixed_gray,
            inputImage=moving_gray,
            warpMatrix=warp_matrix,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        # 如果 ECC 失败，返回原图（不做变换）
        return moving

    # 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
    h, w = fixed.shape[:2]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # 单应变换（透视变换）
        registered = cv2.warpPerspective(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # 仿射变换
        registered = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return registered


def build_fit_mask(env: np.ndarray, mix: np.ndarray, diff_thresh: int = 12) -> np.ndarray:
    """
    找出两张图片中“变化很小”的区域，这些区域用来做亮度匹配，排除棋盘格区域。
    
    生成一个掩膜（mask），标记出哪些像素可以用来做亮度匹配。只选择那些“关灯图和开灯图几乎一样”的区域，
    避开投影棋盘格所在的地方。如果直接用整张图做亮度匹配，棋盘格的亮暗变化会被当成“亮度差异”，
    导致匹配结果不准确。只有那些“没有投影图案”的稳定区域，才能真实反映相机自动曝光带来的整体亮度变化。
    
    原理：
        1. 计算两张图的绝对差值，转成灰度
        2. 找出差值小于阈值的像素（这些地方两张图几乎一样，说明没有投影图案）
        3. 排除过亮或过暗的像素（避免高光或阴影影响匹配）
        4. 用形态学开运算（先腐蚀后膨胀）清理小噪声，得到干净的掩膜
    
    参数：
        env: 对齐后的关灯图
        mix: 开灯图
        diff_thresh: 差分阈值，默认 12。如果两张图在某个像素的差异小于这个值，
                     就认为这个像素是“稳定区域”（没有投影图案）。
    
    返回：
        一个二值掩膜（uint8），255 表示“可以用这个区域做匹配”，0 表示“不能用”。
    """
    # 计算两张图像的绝对差值
    diff = cv2.absdiff(mix, env)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 找出差值小于阈值的像素（稳定区域，没有投影图案）
    mask = (diff_gray < diff_thresh).astype(np.uint8) * 255
    
    # 排除过亮或过暗的像素（避免高光或阴影影响匹配）
    sat = (np.max(mix, axis=2) < 250) & (np.max(env, axis=2) < 250)
    mask[sat == 0] = 0
    
    # 形态学开运算：先腐蚀后膨胀，去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def robust_fit_beta_gamma(env_ch: np.ndarray, mix_ch: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    对单个颜色通道做亮度匹配，找到让“关灯图×比例 + 偏移 ≈ 开灯图”的两个参数。
    
    计算两个参数 beta（比例）和 gamma（偏移），使得：开灯图 ≈ beta × 关灯图 + gamma，
    这样就能补偿相机自动曝光带来的整体亮度差异。相机在拍摄两张图时，可能会自动调整曝光、增益或白平衡，
    导致整体亮度不同。如果直接相减，会残留大片“伪差异”（其实是亮度差异，不是投影图案）。
    通过这个线性匹配，先把两张图拉到同一“亮度尺度”，再相减就干净多了。
    
    原理：
        1. 从掩膜区域提取像素值，用最小二乘法拟合线性关系
        2. 计算拟合残差，找出那些“偏离太远”的异常点
        3. 使用 80 分位数作为阈值，筛选内点
        4. 剔除异常点后重新拟合，得到更稳健的参数
        5. 把参数限制在合理范围内（beta: 0.7-1.3, gamma: -30-30），避免极端值
        6. 如果有效像素太少（<100），返回默认值（beta=1.0, gamma=0），不做匹配
    
    参数：
        env_ch: 关灯图的单个颜色通道（例如红色通道）
        mix_ch: 开灯图的同一个颜色通道
        mask:   掩膜，标记哪些像素可以用来做匹配（255=可用，0=不可用）
    
    返回：
        (beta, gamma) 两个浮点数：
        - beta: 比例系数，通常在 0.7-1.3 之间（如果两张图亮度差不多，beta 接近 1.0）
        - gamma: 偏移量，通常在 -30 到 30 之间（如果两张图亮度一样，gamma 接近 0）
    """
    # 从掩膜区域提取有效像素值
    m = mask.astype(bool)
    x = env_ch[m].astype(np.float32).reshape(-1, 1)  # 关灯图通道值
    y = mix_ch[m].astype(np.float32).reshape(-1, 1)  # 开灯图通道值
    
    # 如果有效样本太少，返回默认值（不做匹配）
    if x.size < 100:
        return 1.0, 0.0
    
    # 构建线性方程组：y = beta * x + gamma
    # X = [x, 1]，用于最小二乘拟合
    X = np.hstack([x, np.ones_like(x)])
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta = float(theta[0, 0])  # 比例系数（提取标量值）
    gamma = float(theta[1, 0])  # 偏移量（提取标量值）
    
    # 计算拟合残差，用于识别异常点
    residuals = (y - (beta * x + gamma)).ravel()
    abs_res = np.abs(residuals)
    
    # 使用 80 分位数作为阈值，筛选内点
    thr = np.percentile(abs_res, 80)
    inliers = abs_res <= thr
    
    # 仅使用内点重新拟合，提高鲁棒性
    Xin = X[inliers]
    yin = y[inliers]
    if Xin.shape[0] >= 50:
        theta2, _, _, _ = np.linalg.lstsq(Xin, yin, rcond=None)
        beta = float(theta2[0, 0])  # 提取标量值
        gamma = float(theta2[1, 0])  # 提取标量值
    
    # 限制参数在合理范围内，避免极端值
    beta = float(np.clip(beta, 0.7, 1.3))
    gamma = float(np.clip(gamma, -30.0, 30.0))
    
    return beta, gamma


def channelwise_compensated_diff(env: np.ndarray, mix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对三个颜色通道（蓝、绿、红）分别做亮度匹配后相减，得到“纯投影反射”。
    
    计算：纯投影反射 = 开灯图 - (beta × 关灯图 + gamma)，对每个颜色通道分别计算，最后合成一张彩色图。
    这是整个流程的核心步骤。通过相减，把“环境反射”去掉，只保留“投影带来的反射”。
    但直接相减会有问题（亮度不匹配），所以先做亮度匹配再相减。
    
    原理：
        1. 对每个颜色通道（B、G、R）分别调用 robust_fit_beta_gamma 找到 beta、gamma
        2. 用公式：纯投影 = mix - (beta × env + gamma) 计算每个通道
        3. 把结果限制在 0-255 范围内（避免负值或溢出）
        4. 合成三通道，得到最终的彩色“纯投影反射图”
    
    参数：
        env: 对齐后的关灯图（BGR 三通道）
        mix: 开灯图（BGR 三通道）
        mask: 拟合掩膜（用来找 beta、gamma）
    
    返回：
        一张彩色图（BGR），表示“纯投影反射”。像素值被限制在 0-255 之间。
    """
    # 初始化输出数组（三通道）
    diff = np.zeros_like(mix, dtype=np.float32)
    
    # 对每个颜色通道（B、G、R）分别处理
    for c in range(3):
        # 计算该通道的亮度匹配参数
        beta, gamma = robust_fit_beta_gamma(env[:, :, c], mix[:, :, c], mask)
        
        # 补偿相减：纯投影反射 = 开灯图 - (beta × 关灯图 + gamma)
        ch = mix[:, :, c].astype(np.float32) - (beta * env[:, :, c].astype(np.float32) + gamma)
        diff[:, :, c] = ch
    
    # 限制像素值在 [0, 255] 范围内，并转换为 uint8
    return np.clip(diff, 0, 255).astype(np.uint8)


def build_chess_mask_from_proj(r_proj_gray: np.ndarray, block_size: int = 31, C: int = -5, min_area: int = 200) -> np.ndarray:
    """
    根据“纯投影反射”的灰度图，自动找出棋盘格所在的大致区域。
    
    生成一个掩膜，标记出棋盘格覆盖的区域。这样后续处理时，只在棋盘格区域内做背景抑制，不会影响其他区域。
    棋盘格区域里有缓慢变化的背景反射（比如天空、建筑），会影响角点检测。但非棋盘格区域不需要处理，
    所以先找出棋盘格在哪里，只处理那一块。
    
    原理：
        1. 先用高斯模糊去噪，让图片更平滑
        2. 用自适应阈值把图片二值化（分成黑白两部分），棋盘格的亮暗条纹会很明显
        3. 用形态学闭运算（先膨胀后腐蚀）填补小孔洞，让棋盘格区域更完整
        4. 找出所有连通域，删除面积太小的（这些是噪声）
        5. 适度膨胀掩膜，确保覆盖到方格内部，不会漏掉边缘
    
    参数：
        r_proj_gray: 纯投影反射的灰度图（已经做过对齐和相减）
        block_size: 自适应阈值的块大小，默认 31（必须是奇数）
        C: 阈值常数，默认 -5（负值更容易保留亮暗条纹）
        min_area: 最小连通域面积，默认 200（小于这个面积的区域会被当成噪声删除）
    
    返回：
        一个二值掩膜（uint8），255 表示“这里是棋盘格区域”，0 表示“不是”。
    """
    # 高斯模糊去噪，使图像更平滑
    blur = cv2.GaussianBlur(r_proj_gray, (5, 5), 0)
    
    # 自适应阈值二值化：将图像分为黑白两部分，突出棋盘格的亮暗条纹
    # block_size 必须是奇数，使用按位或确保为奇数
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, block_size | 1), C)
    
    # 形态学闭运算：先膨胀后腐蚀，填补小孔洞，使棋盘格区域更完整
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 连通域分析：找出所有独立的区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    # 筛选出面积大于阈值的连通域（去除小噪声）
    mask = np.zeros_like(bw)
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    
    # 适度膨胀掩膜，确保覆盖到方格内部，不会漏掉边缘
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel2, iterations=1)
    
    return mask


def suppress_background_in_mask(r_proj_gray: np.ndarray, mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    在棋盘格区域内，去掉缓慢变化的背景亮度，只保留细节纹理，让角点更突出。
    
    使用“高斯高通滤波”，把图片分解成“低频（缓慢变化）”和“高频（细节纹理）”，然后去掉低频部分，只保留高频。
    这样棋盘格的角点会更清晰，而背景反射会被抑制。棋盘格区域里除了投影图案，还有缓慢变化的背景反射（比如天空、建筑倒影）。
    这些背景会干扰角点检测。通过高通滤波，把缓慢变化的背景去掉，只保留棋盘格的快速变化（黑白交替），角点就会更明显。
    
    原理：
        1. 确保核大小为奇数（高斯模糊要求）
        2. 用大核高斯模糊得到“低频分量”（缓慢变化的背景）
        3. 用原图减去低频分量，得到“高频分量”（细节纹理，包括棋盘格）
        4. 只在掩膜标记的区域内用高频分量替换原图，其他区域保持原样
        5. 把结果限制在 0-255 范围内
    
    参数：
        r_proj_gray: 纯投影反射的灰度图
        mask: 棋盘格掩膜（255=棋盘格区域，0=非棋盘格区域）
        ksize: 高斯模糊的核大小，默认 51（必须是奇数）。越大，去掉的“低频”越多，
               但太大可能会影响角点附近的细节。
    
    返回：
        处理后的灰度图。只在掩膜标记的区域内做了背景抑制，其他区域保持原样。
    """
    # 确保核大小为奇数（高斯模糊要求）
    ksize = max(3, ksize | 1)
    
    # 高斯模糊得到低频分量（缓慢变化的背景）
    low = cv2.GaussianBlur(r_proj_gray, (ksize, ksize), 0)
    
    # 原图减去低频分量，得到高频分量（细节纹理）
    filtered = cv2.subtract(r_proj_gray, low)
    
    # 复制原图作为输出，使用 int16 类型以支持负值
    out = r_proj_gray.copy().astype(np.int16)
    
    # 仅在掩膜标记的区域内用高频分量替换原图
    out[mask > 0] = filtered[mask > 0].astype(np.int16)
    
    # 限制像素值在 [0, 255] 范围内，并转换为 uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def find_input_images(data_dir: str) -> Tuple[str, str]:
    """
    在 data 目录里查找文件名包含 “env” 和 “mix” 的图片。
    
    两张图片必须按照该命名规则提供，缺少任意一侧即报错，不再使用模糊匹配或排序兜底。
    """
    # 支持的图片格式
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    
    # 扫描目录，收集所有图片文件
    for e in exts:
        files.extend(glob.glob(os.path.join(data_dir, e)))
    
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到图片")
    
    env_candidates = sorted(
        [f for f in files if "env" in os.path.basename(f).lower()])
    mix_candidates = sorted(
        [f for f in files if "mix" in os.path.basename(f).lower()])
    
    if not env_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'env' 的图片")
    if not mix_candidates:
        raise FileNotFoundError("data 目录中未找到文件名包含 'mix' 的图片")
    
    return env_candidates[0], mix_candidates[0]


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
        10. 保存结果：result_detect.png（用于角点检测的灰度图）
    
    参数：
        无（所有参数通过函数调用时的默认值设置）
    
    返回：
        无（结果保存到 result 目录）
    
    输出说明：
        - result_detect.png：这是给算法用的，已经做了对齐、亮度匹配、相减和背景抑制，
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

    # ========== 保存输出结果 ==========
    # 保存检测用灰度图：用于角点检测和三维重建
    detect_out = os.path.join(out_dir, "result_detect.png")
    safe_imwrite(detect_out, r_proj_hf)
    
    # 保存棋盘格掩膜：用于后续角点检测时限制搜索范围
    mask_out = os.path.join(out_dir, "result_mask.png")
    safe_imwrite(mask_out, chess_mask)
    
    print("\n[完成] 文件已保存：")
    print(f"  棋盘格灰度: {detect_out}")
    print(f"  棋盘格掩膜: {mask_out}")
    print("=== 处理结束 ===")


if __name__ == "__main__":
    main()

