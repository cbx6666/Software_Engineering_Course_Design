"""
亮度匹配相关函数
"""
from typing import Tuple
import cv2
import numpy as np

try:
    from ...config import DEFAULT_CONFIG, ProjectionDiffConfig
    from ...logging_utils import get_logger
except ImportError:
    from config import DEFAULT_CONFIG, ProjectionDiffConfig
    from logging_utils import get_logger


logger = get_logger("projection.brightness")


def _fit_line_from_samples(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = beta * x + gamma without building a large design matrix."""
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denom = float(np.dot(dx, dx))
    if denom <= 1e-12:
        return 1.0, y_mean - x_mean

    beta = float(np.dot(dx, dy) / denom)
    gamma = y_mean - beta * x_mean
    return beta, gamma


def _odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_fit_mask(
    env: np.ndarray,
    mix: np.ndarray,
    diff_thresh: int = None,
    config: ProjectionDiffConfig = None,
) -> np.ndarray:
    """
    找出两张图片中"变化很小"的区域，这些区域用来做亮度匹配，排除棋盘格区域。
    
    生成一个掩膜（mask），标记出哪些像素可以用来做亮度匹配。只选择那些"关灯图和开灯图几乎一样"的区域，
    避开投影棋盘格所在的地方。如果直接用整张图做亮度匹配，棋盘格的亮暗变化会被当成"亮度差异"，
    导致匹配结果不准确。只有那些"没有投影图案"的稳定区域，才能真实反映相机自动曝光带来的整体亮度变化。
    
    原理：
        1. 计算两张图的绝对差值，转成灰度
        2. 找出差值小于阈值的像素（这些地方两张图几乎一样，说明没有投影图案）
        3. 排除过亮或过暗的像素（避免高光或阴影影响匹配）
        4. 用形态学开运算（先腐蚀后膨胀）清理小噪声，得到干净的掩膜
    
    参数：
        env: 对齐后的关灯图
        mix: 开灯图
        diff_thresh: 差分阈值，默认 12。如果两张图在某个像素的差异小于这个值，
                     就认为这个像素是"稳定区域"（没有投影图案）。
    
    返回：
        一个二值掩膜（uint8），255 表示"可以用这个区域做匹配"，0 表示"不能用"。
    """
    config = config or DEFAULT_CONFIG.projection
    base_thresh = config.diff_thresh if diff_thresh is None else diff_thresh

    # 计算两张图像的绝对差值
    diff = cv2.absdiff(mix, env)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    med = float(np.median(diff_gray))
    mad = float(np.median(np.abs(diff_gray.astype(np.float32) - med)))
    robust_thresh = med + config.diff_robust_scale * max(mad, 1.0)
    percentile_thresh = float(np.percentile(diff_gray, config.diff_percentile))
    adaptive_thresh = min(max(base_thresh, robust_thresh, percentile_thresh), config.max_diff_thresh)

    # 找出差值小于阈值的像素（稳定区域，没有投影图案）
    mask = (diff_gray <= adaptive_thresh).astype(np.uint8) * 255
    
    # 排除过亮或过暗的像素（避免高光或阴影影响匹配）
    mix_gray = cv2.cvtColor(mix, cv2.COLOR_BGR2GRAY)
    env_gray = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
    valid_intensity = (
        (mix_gray > config.intensity_low)
        & (env_gray > config.intensity_low)
        & (mix_gray < config.intensity_high)
        & (env_gray < config.intensity_high)
        & (np.max(mix, axis=2) < config.intensity_high)
        & (np.max(env, axis=2) < config.intensity_high)
    )
    mask[~valid_intensity] = 0

    if config.fit_mask_open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _odd_kernel(config.fit_mask_open_kernel), iterations=1)
    if config.fit_mask_close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _odd_kernel(config.fit_mask_close_kernel), iterations=1)

    fit_ratio = float(np.count_nonzero(mask)) / mask.size
    if fit_ratio < config.min_fit_ratio:
        logger.warning(
            "fit mask is too small (%.3f), falling back to valid-intensity mask",
            fit_ratio,
        )
        mask = valid_intensity.astype(np.uint8) * 255
    
    return mask


def robust_fit_beta_gamma(
    env_ch: np.ndarray,
    mix_ch: np.ndarray,
    mask: np.ndarray,
    config: ProjectionDiffConfig = None,
) -> Tuple[float, float]:
    """
    对单个颜色通道做亮度匹配，找到让"关灯图×比例 + 偏移 ≈ 开灯图"的两个参数。
    
    计算两个参数 beta（比例）和 gamma（偏移），使得：开灯图 ≈ beta × 关灯图 + gamma，
    这样就能补偿相机自动曝光带来的整体亮度差异。相机在拍摄两张图时，可能会自动调整曝光、增益或白平衡，
    导致整体亮度不同。如果直接相减，会残留大片"伪差异"（其实是亮度差异，不是投影图案）。
    通过这个线性匹配，先把两张图拉到同一"亮度尺度"，再相减就干净多了。
    
    原理：
        1. 从掩膜区域提取像素值，用最小二乘法拟合线性关系
        2. 计算拟合残差，找出那些"偏离太远"的异常点
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
    config = config or DEFAULT_CONFIG.projection

    # 从掩膜区域提取有效像素值
    m = mask.astype(bool)
    x = env_ch[m].astype(np.float32, copy=False).ravel()  # 关灯图通道值
    y = mix_ch[m].astype(np.float32, copy=False).ravel()  # 开灯图通道值
    
    # 如果有效样本太少，返回默认值（不做匹配）
    if x.size < config.min_fit_samples:
        if x.size > 0:
            gamma = float(np.median(y - x))
            gamma = float(np.clip(gamma, *config.gamma_range))
            return 1.0, gamma
        return 1.0, 0.0
    
    beta, gamma = _fit_line_from_samples(x, y)
    
    # 计算拟合残差，用于识别异常点
    residuals = y - (beta * x + gamma)
    abs_res = np.abs(residuals)
    
    # 使用 80 分位数作为阈值，筛选内点
    thr = np.percentile(abs_res, config.residual_percentile)
    inliers = abs_res <= thr
    
    # 仅使用内点重新拟合，提高鲁棒性
    if np.count_nonzero(inliers) >= config.min_refit_samples:
        beta, gamma = _fit_line_from_samples(x[inliers], y[inliers])
    
    # 限制参数在合理范围内，避免极端值
    beta = float(np.clip(beta, *config.beta_range))
    gamma = float(np.clip(gamma, *config.gamma_range))
    
    return beta, gamma


def channelwise_compensated_diff(
    env: np.ndarray,
    mix: np.ndarray,
    mask: np.ndarray,
    config: ProjectionDiffConfig = None,
) -> np.ndarray:
    """
    对三个颜色通道（蓝、绿、红）分别做亮度匹配后相减，得到"纯投影反射"。
    
    计算：纯投影反射 = 开灯图 - (beta × 关灯图 + gamma)，对每个颜色通道分别计算，最后合成一张彩色图。
    这是整个流程的核心步骤。通过相减，把"环境反射"去掉，只保留"投影带来的反射"。
    但直接相减会有问题（亮度不匹配），所以先做亮度匹配再相减。
    
    原理：
        1. 对每个颜色通道（B、G、R）分别调用 robust_fit_beta_gamma 找到 beta、gamma
        2. 用公式：纯投影 = mix - (beta × env + gamma) 计算每个通道
        3. 把结果限制在 0-255 范围内（避免负值或溢出）
        4. 合成三通道，得到最终的彩色"纯投影反射图"
    
    参数：
        env: 对齐后的关灯图（BGR 三通道）
        mix: 开灯图（BGR 三通道）
        mask: 拟合掩膜（用来找 beta、gamma）
    
    返回：
        一张彩色图（BGR），表示"纯投影反射"。像素值被限制在 0-255 之间。
    """
    config = config or DEFAULT_CONFIG.projection
    env_f = env.astype(np.float32, copy=False)
    mix_f = mix.astype(np.float32, copy=False)

    # 初始化输出数组（三通道）
    diff = np.empty_like(mix_f)
    
    # 对每个颜色通道（B、G、R）分别处理
    for c in range(3):
        # 计算该通道的亮度匹配参数
        beta, gamma = robust_fit_beta_gamma(env_f[:, :, c], mix_f[:, :, c], mask, config=config)
        
        # 补偿相减：纯投影反射 = 开灯图 - (beta × 关灯图 + gamma)
        ch = mix_f[:, :, c] - (beta * env_f[:, :, c] + gamma)
        diff[:, :, c] = ch
    
    # 限制像素值在 [0, 255] 范围内，并转换为 uint8
    return np.clip(diff, 0, 255).astype(np.uint8)

