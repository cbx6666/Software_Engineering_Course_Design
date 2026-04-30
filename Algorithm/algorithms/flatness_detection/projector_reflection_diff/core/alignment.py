"""
图像对齐相关函数
"""
from typing import Tuple
import cv2
import numpy as np

try:
    from ...config import DEFAULT_CONFIG, ProjectionDiffConfig
    from ...logging_utils import get_logger
except ImportError:  # Compatible with detect.py adding flatness_detection to sys.path.
    from config import DEFAULT_CONFIG, ProjectionDiffConfig
    from logging_utils import get_logger


logger = get_logger("projection.alignment")


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


def _warp_mode_and_matrix(motion_model: str) -> Tuple[int, np.ndarray]:
    if motion_model == "homography":
        return cv2.MOTION_HOMOGRAPHY, np.eye(3, 3, dtype=np.float32)
    if motion_model == "translation":
        return cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)
    return cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)


def _prepare_ecc_gray(img: np.ndarray, blur_kernel: int) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if blur_kernel > 1:
        blur_kernel = max(3, blur_kernel | 1)
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    gray = gray.astype(np.float32, copy=False)
    cv2.normalize(gray, gray, 0.0, 1.0, cv2.NORM_MINMAX)
    return gray


def _scale_warp_to_full(warp_matrix: np.ndarray, warp_mode: int, scale_x: float, scale_y: float) -> np.ndarray:
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
        scale_inv = np.array([[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]], dtype=np.float32)
        return scale_inv @ warp_matrix @ scale

    warp_matrix = warp_matrix.copy()
    warp_matrix[0, 2] *= scale_x
    warp_matrix[1, 2] *= scale_y
    return warp_matrix


def _run_ecc(
    moving: np.ndarray,
    fixed: np.ndarray,
    motion_model: str,
    config: ProjectionDiffConfig,
    fast_mode: bool,
    max_iterations: int,
    eps: float,
) -> Tuple[int, np.ndarray]:
    h, w = fixed.shape[:2]
    warp_mode, warp_matrix = _warp_mode_and_matrix(motion_model)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)

    if fast_mode and min(h, w) > config.ecc_downsample_min_size:
        scale = config.ecc_downsample_min_size / float(min(h, w))
        h_small = max(2, int(round(h * scale)))
        w_small = max(2, int(round(w * scale)))
        fixed_small = cv2.resize(fixed, (w_small, h_small), interpolation=cv2.INTER_AREA)
        moving_small = cv2.resize(moving, (w_small, h_small), interpolation=cv2.INTER_AREA)

        fixed_gray = _prepare_ecc_gray(fixed_small, config.ecc_blur_kernel)
        moving_gray = _prepare_ecc_gray(moving_small, config.ecc_blur_kernel)
        _, warp_matrix = cv2.findTransformECC(
            fixed_gray,
            moving_gray,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=config.ecc_gauss_filter_size,
        )

        warp_matrix = _scale_warp_to_full(warp_matrix, warp_mode, w / w_small, h / h_small)

        if config.ecc_refine_iterations > 0:
            refine_criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.ecc_refine_iterations,
                config.ecc_refine_eps,
            )
            fixed_gray_full = _prepare_ecc_gray(fixed, config.ecc_blur_kernel)
            moving_gray_full = _prepare_ecc_gray(moving, config.ecc_blur_kernel)
            _, warp_matrix = cv2.findTransformECC(
                fixed_gray_full,
                moving_gray_full,
                warp_matrix,
                warp_mode,
                refine_criteria,
                inputMask=None,
                gaussFiltSize=config.ecc_gauss_filter_size,
            )
    else:
        fixed_gray = _prepare_ecc_gray(fixed, config.ecc_blur_kernel)
        moving_gray = _prepare_ecc_gray(moving, config.ecc_blur_kernel)
        _, warp_matrix = cv2.findTransformECC(
            fixed_gray,
            moving_gray,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=config.ecc_gauss_filter_size,
        )

    return warp_mode, warp_matrix


def _apply_warp(moving: np.ndarray, fixed_shape: Tuple[int, int], warp_mode: int, warp_matrix: np.ndarray) -> np.ndarray:
    h, w = fixed_shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(
            moving,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return cv2.warpAffine(
        moving,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def ecc_register(
    moving: np.ndarray,
    fixed: np.ndarray,
    motion_model: str = "homography",
    fast_mode: bool = True,
    max_iterations: int = 50,
    eps: float = 1e-4,
    config: ProjectionDiffConfig = None,
) -> np.ndarray:
    """
    把关灯图的位置"对齐"到开灯图，消除相机轻微移动造成的错位。
    
    即使拍摄时相机有轻微移动或角度变化，也能让两张图片的像素点"对得上"，这样后续相减时不会出现重影或模糊。
    无人机或手持拍摄时，很难保证两张图片完全对齐。如果直接相减，错位的地方会出现"重影"，影响结果质量。
    对齐后，同一位置的像素才真正对应同一块玻璃区域。
    
    优化策略（fast_mode=True）：
        1. 多级优化：先在下采样图像（~800像素）上快速找到初始对齐（约3-5倍加速）
        2. 精细优化：使用初始结果在原图上进行少量迭代优化（提高精度）
        3. 降低迭代次数：从200次降至50次（快速模式）或保持200次（标准模式）
        4. 放宽精度阈值：从1e-6放宽至1e-4（快速模式）或保持1e-6（标准模式）
    
    原理：
        1. （快速模式）对图像进行下采样，在小尺寸上快速计算变换矩阵
        2. 将变换矩阵正确缩放到原图尺寸
        3. （可选）在原图上进行精细优化
        4. 应用变换矩阵，将 moving 图像对齐到 fixed 图像的坐标系
    
    参数：
        moving: 需要被对齐的图片（关灯图）
        fixed:  参考图片（开灯图），moving 会被对齐到这个图片的坐标系
        motion_model: 对齐模型，默认 "homography"（单应性，能处理轻微视角变化）
                      也可以选 "affine"（仿射，只能处理平移、旋转、缩放）
        fast_mode: 是否使用快速模式（多级优化），默认 True
        max_iterations: 最大迭代次数，默认 50（快速模式）或 200（标准模式）
        eps: 精度阈值，默认 1e-4（快速模式）或 1e-6（标准模式）
    
    返回：
        对齐后的关灯图，尺寸和 fixed 一样。
    """
    config = config or DEFAULT_CONFIG.projection
    h, w = fixed.shape[:2]
    models = []
    if motion_model:
        models.append(motion_model)
    models.extend(model for model in config.alignment_models if model not in models)

    for model in models:
        if model == "identity":
            break
        try:
            warp_mode, warp_matrix = _run_ecc(
                moving,
                fixed,
                model,
                config,
                fast_mode=fast_mode,
                max_iterations=max_iterations or config.ecc_max_iterations,
                eps=eps or config.ecc_eps,
            )
            logger.info("ECC alignment succeeded with %s model", model)
            return _apply_warp(moving, (h, w), warp_mode, warp_matrix)
        except cv2.error as exc:
            logger.warning("ECC alignment failed with %s model: %s", model, exc)
        except Exception as exc:
            logger.warning("ECC alignment failed with %s model: %s", model, exc)

    logger.warning("ECC alignment fell back to identity transform")
    return resize_to_match(moving, (h, w))
