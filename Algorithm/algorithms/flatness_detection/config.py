"""平整度检测流程的统一配置。"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DebugOutputConfig:
    """调试中间结果输出配置。"""

    enabled: bool = False
    output_dir_name: str = "debug"


@dataclass
class ProjectionDiffConfig:
    """投影反射差分阶段的配置。"""

    alignment_models: Tuple[str, ...] = ("homography", "affine", "translation")
    ecc_downsample_min_size: int = 800
    ecc_max_iterations: int = 50
    ecc_eps: float = 1e-4
    ecc_refine_iterations: int = 15
    ecc_refine_eps: float = 1e-5
    ecc_gauss_filter_size: int = 5
    ecc_blur_kernel: int = 5
    diff_thresh: int = 12
    diff_robust_scale: float = 2.5
    diff_percentile: float = 65.0
    max_diff_thresh: int = 45
    intensity_low: int = 5
    intensity_high: int = 245
    fit_mask_open_kernel: int = 5
    fit_mask_close_kernel: int = 7
    min_fit_ratio: float = 0.01
    min_fit_samples: int = 100
    min_refit_samples: int = 50
    residual_percentile: float = 80.0
    beta_range: Tuple[float, float] = (0.7, 1.3)
    gamma_range: Tuple[float, float] = (-30.0, 30.0)
    adaptive_block_size: int = 31
    adaptive_c: int = -5
    chess_min_area: int = 200
    chess_keep_regions: int = 3
    chess_mask_max_ratio: float = 0.35
    chess_candidate_expand_kernel: int = 11
    chess_open_kernel: int = 3
    chess_close_kernel: int = 7
    chess_dilate_kernel: int = 7
    min_chess_mask_ratio: float = 0.001
    texture_blur_kernel: int = 9
    texture_window_size: int = 41
    texture_percentile: float = 92.0
    texture_open_kernel: int = 5
    texture_close_kernel: int = 9
    texture_min_score: float = 8.0
    geometry_min_gradient_balance: float = 0.18
    geometry_min_periodicity: float = 1.15
    geometry_min_axis_contrast: float = 2.0
    background_ksize: int = 51
    contrast_clip_limit: float = 2.0
    contrast_tile_grid_size: Tuple[int, int] = (8, 8)
    debug: DebugOutputConfig = field(default_factory=DebugOutputConfig)


@dataclass
class CornerDetectionConfig:
    """棋盘格角点检测阶段的配置。"""

    chessboard_size: Tuple[int, int] = (7, 10)
    allow_partial: bool = True
    min_corner_ratio: float = 0.5
    scales: Tuple[float, ...] = (1.0, 1.25, 1.5, 0.8)
    preprocess_modes: Tuple[str, ...] = ("original", "clahe", "blur", "threshold", "sharpen")
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    blur_kernel: int = 3
    threshold_block_size: int = 31
    threshold_c: int = 3
    min_area_coverage: float = 0.003
    min_duplicate_distance: float = 1.0
    max_spacing_ratio: float = 6.0
    min_spacing: float = 2.0
    subpix_window: Tuple[int, int] = (11, 11)
    subpix_zero_zone: Tuple[int, int] = (-1, -1)
    subpix_max_iter: int = 30
    subpix_eps: float = 0.001
    crop_padding: int = 50
    crop_min_area: int = 5000
    crop_component_min_fill_ratio: float = 0.08
    crop_aspect_ratio_range: Tuple[float, float] = (0.2, 5.0)
    debug: DebugOutputConfig = field(default_factory=DebugOutputConfig)


@dataclass
class CornerMatchingConfig:
    """左右角点匹配阶段的配置。"""

    min_match_ratio: float = 0.5
    min_matched_points: int = 8
    require_positive_disparity: bool = False
    min_positive_disparity: float = 1e-6
    disparity_mad_thresh: float = 3.5
    y_diff_spacing_factor: float = 1.5
    max_outlier_ratio: float = 0.35
    strict_geometry: bool = False
    filter_outliers: bool = True
    save_quality_json: bool = True
    debug: DebugOutputConfig = field(default_factory=DebugOutputConfig)


@dataclass
class LoggingConfig:
    """日志输出配置。"""

    level: str = "INFO"


@dataclass
class FlatnessConfig:
    """平整度检测总配置。"""

    projection: ProjectionDiffConfig = field(default_factory=ProjectionDiffConfig)
    corner_detection: CornerDetectionConfig = field(default_factory=CornerDetectionConfig)
    corner_matching: CornerMatchingConfig = field(default_factory=CornerMatchingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


DEFAULT_CONFIG = FlatnessConfig()
