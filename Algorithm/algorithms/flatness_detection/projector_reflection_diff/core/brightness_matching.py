"""
亮度匹配相关函数
"""
from typing import Tuple
import cv2
import numpy as np


def build_fit_mask(env: np.ndarray, mix: np.ndarray, diff_thresh: int = 12) -> np.ndarray:
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

