# 平整度检测算法说明

## 1. 模块定位

本模块用于玻璃幕墙平整度检测，入口为：

```python
algorithms.flatness_detection.detect.main(data_dir, result_dir, config=None)
```

默认输入为四张图像：

```text
left_env
left_mix
right_env
right_mix
```

其中：

- `env` 表示未投影棋盘格时的环境反射图。
- `mix` 表示投影棋盘格后的混合反射图。
- `left` 和 `right` 分别表示双目左、右相机图像。

主流程保持旧接口兼容，不要求调用方修改 FastAPI 服务层或前端请求结构。

## 2. 文件结构

```text
flatness_detection/
  detect.py                         主流程入口
  config.py                         统一配置
  errors.py                         领域异常
  logging_utils.py                  日志工具
  pipeline_helpers.py               输出文件保存与输入文件查找
  projector_reflection_diff/        投影反射差分
  stereo_corner_matching/           角点检测与左右匹配
  pointcloud_gen/                   点云生成和平整度计算
```

核心子模块：

```text
projector_reflection_diff/core/alignment.py
projector_reflection_diff/core/brightness_matching.py
projector_reflection_diff/core/chessboard_processing.py
projector_reflection_diff/core/chessboard_texture.py
projector_reflection_diff/core/chessboard_candidate.py
projector_reflection_diff/core/chessboard_mask_utils.py

stereo_corner_matching/core/image_processing.py
stereo_corner_matching/core/corner_detection.py
stereo_corner_matching/core/corner_matching.py

pointcloud_gen/core/stereo_process.py
```

## 3. 输出文件

主流程输出到 `result_dir`：

```text
left_detect.png
right_detect.png
left_mask.png
right_mask.png
corners_left.json
corners_right.json
match_quality.json
flatness_metrics.json
pointcloud_data.json
flatness.png
```

兼容性约束：

- `corners_left.json` 和 `corners_right.json` 仍为二维点数组：

```json
[[x1, y1], [x2, y2]]
```

- `pointcloud_data.json` 面向前端展示，当前导出：

```json
{
  "projected_points": [[x, y, z]],
  "projected_dists": [z]
}
```

`projected_points` 单位为米，坐标位于拟合平面的局部坐标系中；`projected_dists` 为相对拟合平面的有符号高度。

## 4. 总体流程

主流程由 `detect.py` 编排：

1. 查找四张输入图像。
2. 分别对左右相机执行投影反射差分，生成棋盘格增强图和掩膜。
3. 在左右增强图上检测棋盘格角点。
4. 对左右角点进行网格化排序和物理点匹配。
5. 保存匹配后的角点 JSON。
6. 根据双目视差恢复三维点。
7. 拟合参考平面并计算平整度指标。
8. 保存点云、指标和可视化结果。

## 5. 投影反射差分

实现位置：

```text
projector_reflection_diff/main.py
projector_reflection_diff/core/
```

### 5.1 图像尺寸统一

`resize_to_match` 将 `env` 图像调整到 `mix` 图像尺寸。后续所有像素级差分、掩膜和角点检测都以 `mix` 尺寸为基准。

### 5.2 图像对齐

`ecc_register` 使用 OpenCV ECC 估计 `env -> mix` 的图像变换。

默认尝试顺序：

```text
homography -> affine -> translation
```

若全部失败，返回尺寸匹配后的原 `env` 图像，相当于 identity fallback。

预处理策略：

- BGR 转灰度。
- 高斯模糊降噪。
- 灰度归一化到 `[0, 1]`。
- 大图先下采样估计变换，再映射回原图。
- 可选少量原图 refine。
- 捕获 `cv2.error` 和普通异常，失败时写 warning 日志。

对齐输出使用：

```text
cv2.warpPerspective
cv2.warpAffine
```

边界策略为 `BORDER_REPLICATE`。

### 5.3 亮度匹配

实现位置：

```text
brightness_matching.py
```

目标是估计：

```text
mix ≈ beta * env_aligned + gamma
```

并计算投影残差：

```text
projection_diff = mix - (beta * env_aligned + gamma)
```

每个 BGR 通道独立拟合 `beta` 和 `gamma`。

拟合掩膜 `mask_fit` 的构建规则：

- 根据 `abs(mix - env)` 估计稳定区域。
- 阈值同时考虑固定阈值、MAD 鲁棒阈值和分位数阈值。
- 排除过亮、过暗和饱和像素。
- 使用 open 和 close 去除小噪声、连接稳定区域。
- 若有效区域过小，退回到亮度有效区域。

拟合方式：

- 使用闭式线性拟合，不构造大规模设计矩阵。
- 先拟合一次，计算残差。
- 根据残差分位数筛选内点。
- 使用内点二次拟合。
- 限制 `beta` 和 `gamma` 在配置范围内。
- 样本不足时退化为中位数偏移或不补偿。

### 5.4 棋盘格掩膜

实现位置：

```text
chessboard_processing.py
chessboard_texture.py
chessboard_candidate.py
chessboard_mask_utils.py
```

掩膜目标不是输出最终检测结果，而是定位棋盘格区域，供角点检测阶段裁剪 ROI。

当前采用三段式：

```text
纹理驱动 -> 候选区域筛选 -> 轻量几何验证
```

#### 5.4.1 纹理驱动

`compute_texture_map` 计算棋盘格纹理响应：

- 局部标准差响应。
- Laplacian 高频边缘密度。
- 两者加权合成纹理热力图。

`build_local_texture_mask` 根据纹理热力图分位数提取高响应区域，并进行形态学 open、close 和连通域过滤。

该策略适配棋盘格占图像比例较小的场景，避免单纯按亮度或面积选择背景高亮区域。

#### 5.4.2 候选区域筛选

`select_best_candidate` 遍历纹理连通域，构造候选区。

候选区构建：

- 对纹理连通域进行膨胀。
- 与自适应阈值掩膜求交或合并。
- close、填洞、膨胀。
- 保留主要连通域。

候选区过滤：

- 面积不能小于 `chess_min_area`。
- 占全图比例不能超过 `chess_mask_max_ratio`。
- 填充率需要合理。

候选区评分考虑：

- 纹理响应均值。
- 填充率。
- 周期性强度。
- 横纵梯度平衡。
- 候选面积平方根。

#### 5.4.3 轻量几何验证

`candidate_geometry_metrics` 计算：

- `gradient_balance`：横纵梯度是否同时存在。
- `periodicity`：行列方向灰度投影是否具有周期性。
- `axis_contrast`：行列方向投影是否有足够对比度。

若存在通过几何验证的候选区，优先选择其中评分最高者；否则选择评分最高候选区作为兼容兜底。

### 5.5 棋盘格增强图

得到 `chess_mask` 后：

1. `suppress_background_in_mask` 在掩膜内部抑制低频背景。
2. `enhance_chessboard_contrast` 在掩膜内部执行 CLAHE。
3. 保存增强图为 `left_detect.png` / `right_detect.png`。
4. 保存掩膜为 `left_mask.png` / `right_mask.png`。

## 6. 棋盘格角点检测

实现位置：

```text
stereo_corner_matching/core/image_processing.py
stereo_corner_matching/core/corner_detection.py
```

### 6.1 ROI 裁剪

`crop_image_by_mask` 使用 `left_mask.png` / `right_mask.png` 定位棋盘格候选框。

裁剪逻辑：

- 将掩膜转为二值图。
- 对连通域进行面积、填充率和长宽比筛选。
- 选择评分最高的候选框。
- 按 `crop_padding` 扩展裁剪区域。
- 返回裁剪图、裁剪掩膜和左上角偏移量。

如果无法定位 ROI，则返回原图和原掩膜。

### 6.2 多策略角点检测

`find_chessboard_corners` 按以下维度组合尝试：

预处理模式：

```text
original
clahe
blur
threshold
sharpen
```

尺度：

```text
1.0
1.25
1.5
0.8
```

检测方法：

```text
findChessboardCornersSB
findChessboardCorners
```

SB 检测优先；如果当前 OpenCV 不支持某些 SB flags，则按 `hasattr` 判断后跳过。传统检测使用：

```text
CALIB_CB_ADAPTIVE_THRESH
CALIB_CB_NORMALIZE_IMAGE
CALIB_CB_FILTER_QUADS
CALIB_CB_PARTIAL_OK
```

其中 `CALIB_CB_PARTIAL_OK` 通过 `getattr` 兼容不同 OpenCV 版本。

### 6.3 角点格式兼容

`_opencv_result` 和 `_normalize_corners` 将 OpenCV 返回值统一为：

```text
(N, 1, 2), dtype=float32
```

兼容：

- `(ok, corners)`。
- 直接返回 corners。
- `None`。
- `(N, 2)`。

### 6.4 质量校验

`validate_corner_quality` 检查：

- 角点数量是否满足 `min_corner_ratio`。
- 角点覆盖范围。
- 重复点距离。
- 最近邻中位距离。
- 完整棋盘格的行列单调性。
- 完整棋盘格的行列间距异常。

默认棋盘格尺寸：

```text
chessboard_size = (7, 10)
```

解释：该值表示内角点数量，实际角点数为 `7 * 10 = 70`。

兼容策略：

- 完整检测到 70 个角点时，不因棋盘格占整图比例小而直接拒绝。
- 部分检测时，覆盖范围过小会作为失败条件。

### 6.5 亚像素精化

`refine_corners` 使用 `cv2.cornerSubPix`。

保护策略：

- 输入先转为灰度 `uint8`。
- 角点先规范化为 `(N, 1, 2)`。
- 捕获 `cv2.error`。
- 失败时返回原始角点，不中断主流程。

## 7. 左右角点匹配

实现位置：

```text
stereo_corner_matching/core/corner_matching.py
```

### 7.1 网格重建

`sort_corners_to_grid` 将角点分配到固定棋盘格网格。

步骤：

1. 使用 SVD 估计角点整体主方向。
2. 根据 `pattern_size` 判断行、列方向。
3. 将角点投影到行轴和列轴。
4. 分别对投影值做一维聚类，得到行列中心。
5. 每个角点分配到最近的网格单元。
6. 若同一网格单元有多个角点，保留投影代价更小者。
7. 缺失网格位置保留 `NaN`。

该策略避免直接按 `y` 分组、按 `x` 排序导致透视变形或缺点时错位。

### 7.2 匹配生成

左右图分别重建网格后，仅保留左右都有效的位置：

```text
both_valid = valid_left & valid_right
```

输出：

```text
matched_left
matched_right
```

其下标一一对应同一个棋盘格物理点。

### 7.3 几何校验

`_validate_and_filter_matches` 检查：

- 匹配点数量。
- 水平视差：

```text
disparity = x_left - x_right
```

- 视差 MAD 离群点。
- 左右点 `y` 坐标差。
- 可选正视差约束。

默认兼容模式：

```text
strict_geometry = False
require_positive_disparity = False
```

含义：

- 几何异常会记录 warning。
- 少量异常点可被剔除。
- 若异常比例过高，默认不整批拒绝旧数据，保留匹配结果并输出质量指标。
- 若需要强约束，可设置 `strict_geometry=True`。

### 7.4 匹配质量输出

`match_quality.json` 包含：

```text
matched_count
initial_matched_count
expected_count
match_ratio
disparity_mean
disparity_std
disparity_min
disparity_max
y_diff_mean
y_diff_std
outlier_count
```

该文件为新增诊断输出，不影响旧流程读取 `corners_left.json` 和 `corners_right.json`。

## 8. 点云生成和平整度计算

实现位置：

```text
pointcloud_gen/core/stereo_process.py
```

### 8.1 视差与深度

由匹配角点计算稀疏视差：

```text
disp = u_left - u_right
```

深度计算：

```text
Z = f * baseline / disp
```

其中：

- `f = K[0, 0]`
- `baseline = 0.265 m`

当前主流程中相机参数为代码内默认值：

```python
K = [
    [800, 0, image_width / 2],
    [0, 800, image_height / 2],
    [0, 0, 1],
]
baseline = 0.265
```

### 8.2 三维反投影

使用相机内参将左图像素点和深度反投影为三维点：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

单位为米。

### 8.3 平面拟合

对有效稀疏三维点执行鲁棒平面拟合。

平面形式：

```text
aX + bY + cZ + d = 0
```

点到平面的有符号距离：

```text
distance = (aX + bY + cZ + d) / sqrt(a^2 + b^2 + c^2)
```

距离单位为米，保存指标时转换为毫米。

### 8.4 稠密点云

当 `densify=True` 时：

1. 使用稀疏角点视差插值生成稠密视差图。
2. 由稠密视差恢复深度。
3. 反投影生成稠密点云。
4. 将稠密点云投影到拟合平面局部坐标系。

稠密点云主要用于可视化和导出，不用于当前平整度核心指标统计。

### 8.5 平整度指标

当前指标来自稀疏测量点到拟合平面的距离：

```text
flatness_range_mm = max(distance_mm) - min(distance_mm)
flatness_rms_mm   = sqrt(mean(distance_mm^2))
flatness_mean_mm  = mean(distance_mm)
flatness_std_mm   = std(distance_mm)
max_mm            = max(distance_mm)
min_mm            = min(distance_mm)
p95_mm            = percentile(abs(distance_mm), 95)
count             = 有效稀疏点数
```

使用稀疏点作为指标来源，是为了避免插值点改变统计结果。

## 9. 配置

统一配置定义在：

```text
config.py
```

总配置：

```python
FlatnessConfig
```

子配置：

```python
ProjectionDiffConfig
CornerDetectionConfig
CornerMatchingConfig
LoggingConfig
DebugOutputConfig
```

### 9.1 ProjectionDiffConfig

主要控制：

- ECC 模型序列。
- ECC 下采样、迭代次数、收敛阈值。
- 亮度拟合掩膜阈值。
- 亮度线性拟合参数范围。
- 棋盘格掩膜面积、连通域数量、最大占比。
- 纹理热力图窗口、分位数和形态学核。
- 候选区域几何验证阈值。
- 对比度增强参数。
- debug 输出开关。

### 9.2 CornerDetectionConfig

主要控制：

- 棋盘格内角点尺寸。
- 是否允许部分检测。
- 最小角点比例。
- 多尺度检测尺度。
- 预处理模式。
- CLAHE、blur、threshold 参数。
- 覆盖率、重复点、间距质量阈值。
- `cornerSubPix` 参数。
- ROI 裁剪 padding、面积、填充率和长宽比。

### 9.3 CornerMatchingConfig

主要控制：

- 最小匹配比例。
- 最小匹配点数。
- 是否要求正视差。
- 视差 MAD 阈值。
- `y` 坐标差阈值。
- 最大异常匹配比例。
- 是否严格几何校验。
- 是否剔除异常点。
- 是否保存 `match_quality.json`。

## 10. 日志与调试输出

日志使用 Python `logging`，默认级别为 `INFO`。

debug 输出由各阶段配置控制：

```python
DebugOutputConfig(enabled=True, output_dir_name="debug")
```

可输出的典型中间结果：

```text
aligned_env
mask_fit
raw_diff
chess_mask
enhanced_chessboard
detected_corners_visualization
matched_corners_visualization
```

默认不启用 debug 输出，避免污染 `result_dir`。

## 11. 异常与失败边界

领域异常：

```text
CornerDetectionError
CornerMatchingError
```

典型失败情况：

- 四图输入缺失。
- 图像无法读取。
- 角点数量不足。
- 部分角点覆盖区域过小。
- 角点重复或间距异常。
- 左右角点有效重叠过少。
- 严格模式下几何校验失败。
- 平面拟合无有效三维点。

兼容策略：

- ECC 失败不直接中断，退化到更简单模型或 identity。
- `cornerSubPix` 失败不直接中断，返回原角点。
- 几何异常默认进入兼容模式，通过 `match_quality.json` 暴露质量风险。

## 12. 测试覆盖

测试文件：

```text
Algorithm/tests/test_flatness_front_flows.py
```

覆盖范围：

- ECC 失败 fallback。
- 输入尺寸不同。
- 明显亮度变化。
- 亮度拟合样本不足兜底。
- 小占比棋盘格掩膜。
- 大平滑高亮区域抑制。
- filled mask ROI 裁剪。
- 正常棋盘格角点检测。
- 低对比度棋盘格角点检测。
- OpenCV 角点返回格式兼容。
- `cornerSubPix` 失败 fallback。
- 完整角点但覆盖率小的兼容。
- 完整左右角点匹配。
- 缺失角点匹配。
- 视差异常点剔除。
- 匹配点过少报错。
- `detect.main(data_dir, result_dir)` 旧入口兼容。

运行：

```powershell
python -m compileall -q Algorithm
python -m unittest discover -s Algorithm\tests -v
```

前端点云展示验证：

```powershell
cd front-end
npm.cmd run build
```

## 13. 工程约束

当前实现保持以下约束：

- 不改变四图输入模式。
- 不改变 `detect.main(data_dir, result_dir)` 主入口。
- 不改变 `corners_left.json` 和 `corners_right.json` 的二维点数组格式。
- 不要求服务层修改返回结构。
- 不引入大型新依赖。
- 优先使用 OpenCV、NumPy、SciPy 等已有依赖。
- 配置集中在 `config.py`，避免新增魔法数字散落在流程代码中。

## 14. 已知限制

1. 相机参数仍使用默认估计值。

当前 `K` 和 `baseline` 不是从真实双目标定文件读取。平整度绝对数值依赖真实内参、外参和基线。若需要测量级精度，必须接入真实标定参数。

2. 稠密点云来自插值。

稠密点云用于可视化，不能等价理解为真实逐像素测量。核心平整度指标当前来自稀疏棋盘格角点。

3. 强反光和遮挡仍可能影响检测。

纹理驱动掩膜可以降低大面积高亮背景误检，但无法完全解决棋盘格严重遮挡、过曝或反射消失的问题。

4. 几何校验默认兼容旧数据。

默认 `strict_geometry=False`，目的是保持旧数据可运行。若用于生产测量，应结合真实标定数据开启更严格的几何约束。

5. 棋盘格规格固定为默认内角点 `(7, 10)`。

若更换投影棋盘格规格，需要同步调整 `CornerDetectionConfig.chessboard_size`，并确认前端和测试数据一致。

