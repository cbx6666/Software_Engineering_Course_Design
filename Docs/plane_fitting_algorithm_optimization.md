# 平面拟合算法代码检查与优化建议

## 检查范围

本次主要检查平整度检测链路中的平面拟合、距离计算、投影展示和指标统计逻辑，涉及文件如下：

- `Algorithm/algorithms/flatness_detection/pointcloud_gen/utils/plane_fit.py`
- `Algorithm/algorithms/flatness_detection/pointcloud_gen/core/stereo_process.py`
- `Algorithm/algorithms/flatness_detection/pointcloud_gen/utils/io_utils.py`
- `Algorithm/algorithms/flatness_detection/detect.py`
- `Algorithm/algorithms/flatness_detection/pointcloud_gen/utils/interp.py`

## 当前实现摘要

当前流程大致为：

1. 通过左右图角点视差恢复稀疏 3D 点云。
2. 用 `mad_mask(pts_valid[:, 2])` 对原始 Z 值做一次异常点过滤。
3. 用 `fit_plane_least_squares()` 拟合 `z = ax + by + c`。
4. 用 `point_plane_signed_distance()` 计算点到该平面的有符号正交距离。
5. 若开启稠密化，则用稀疏视差插值生成全图稠密点云，再计算稠密点到稀疏拟合平面的距离。
6. 平整度指标默认优先使用稠密距离，否则使用稀疏距离。
7. 可视化投影使用 `project_to_plane_normal()` 重新基于稀疏点云做 SVD 平面拟合。

## 主要优化点

### P0: 拟合目标与指标目标不一致

位置：

- `plane_fit.py:3-13`
- `plane_fit.py:15-20`

当前 `fit_plane_least_squares()` 用 `np.linalg.lstsq(A, Z)` 拟合 `z = ax + by + c`，本质是最小化 Z 方向残差。但后续 `point_plane_signed_distance()` 统计的是点到平面的正交距离。

这会带来两个问题：

- 算法拟合目标不是最终指标目标。
- 当平面有明显倾斜或点云坐标尺度变化较大时，Z 残差最小的平面不一定是正交距离意义下的最佳平面。

建议改为总最小二乘或 PCA/SVD 平面拟合，直接最小化正交残差：

```python
centroid = points.mean(axis=0)
centered = points - centroid
_, _, vh = np.linalg.svd(centered, full_matrices=False)
normal = vh[-1]
normal = normal / np.linalg.norm(normal)
offset = -np.dot(normal, centroid)
dist = points @ normal + offset
```

建议保留 `z = ax + by + c` 作为兼容输出，但内部主模型应使用通用平面形式：

```text
normal[0] * x + normal[1] * y + normal[2] * z + offset = 0
```

### P0: 平面表示假设过强

位置：

- `plane_fit.py:4`
- `plane_fit.py:18`

当前模型固定为 `z = ax + by + c`，隐含假设是平面不能接近垂直于 Z 轴。玻璃平整度场景通常接近该假设，但算法层面仍应避免把核心模型绑定到单一坐标表达。

建议：

- 内部统一使用 `normal + offset` 表示平面。
- 只有在 `abs(normal[2]) > eps` 时，才换算兼容的 `(a, b, c)`。
- 若 `abs(normal[2])` 太小，直接返回通用平面，不再强行转换为 `z = ax + by + c`。

### P0: 异常点过滤基于原始 Z，容易误删有效边缘点

位置：

- `stereo_process.py:41-45`
- `outliers.py:3-12`

当前在拟合前用 `pts_valid[:, 2]` 做 MAD 过滤。这个策略只看深度 Z，不看点到平面的残差。若玻璃本身相对相机存在倾斜，边缘区域的 Z 变化可能是正常几何趋势，却会被当作异常值剔除。

建议改为两阶段鲁棒拟合：

1. 第一次用全部有限点做 SVD 平面拟合。
2. 计算所有点到初始平面的正交残差。
3. 对正交残差做 MAD 或分位数过滤。
4. 用内点重新拟合最终平面。
5. 输出内点数、外点数、外点比例和残差阈值。

如果误匹配较多，可进一步引入 RANSAC 或 Huber/IRLS，但优先级低于残差 MAD 二次拟合。

### P0: 可视化投影与指标基准面不一致

位置：

- `stereo_process.py:47-52`
- `io_utils.py:51-86`

指标距离使用 `fit_plane_least_squares(pts_for_fit)` 得到的平面；但 `project_to_plane_normal(pts_sparse)` 又在 `io_utils.py` 中对 `pts_sparse` 重新做了一次 SVD 拟合。也就是说：

- 平整度指标基于一个平面。
- 前端展示的 `projected_points` 和 `projected_dists` 基于另一个平面。

这会导致颜色、正负偏差和统计指标不完全对应。

建议：

- 平面拟合只做一次。
- `project_to_plane_normal()` 接收已拟合的 `normal`、`offset` 和 `centroid`。
- 前端展示的 Z' 或颜色距离直接使用同一个平面的正交距离。

### P0: 有符号距离方向与业务文案可能相反

位置：

- `plane_fit.py:18`
- `front-end/src/components/PointCloud3D.tsx` 中颜色说明
- `Algorithm/app/services/flatness_service.py` 中 max/min 偏差解释

当前距离公式为：

```python
num = a * x + b * y - z + c
```

如果按 `z_plane = ax + by + c` 理解，则点的 `z > z_plane` 时距离为负。因此若业务把更大的 Z 解释为“更高”，当前符号和文案中的“正偏差表示最高点”是相反的。

需要先明确坐标系含义：

- 若希望“高于拟合平面”为正，应使用 `z - z_plane` 或强制法向量朝业务定义的正方向。
- 若相机坐标 Z 表示深度，不应直接把 Z 更大解释为几何高度，应在文档和 UI 中说明正负方向。

建议在平面模型中加入明确的法向量方向规则，例如：

```python
if normal[2] < 0:
    normal = -normal
    offset = -offset
```

然后所有距离、颜色和 max/min 文案都以该规则为准。

### P1: 输入有效性检查不足

位置：

- `plane_fit.py:3-20`
- `stereo_process.py:41-48`
- `io_utils.py:59-66`

当前实现缺少以下保护：

- 点数少于 3 时仍可能进入拟合。
- 点云三点共线或近似共线时没有报错。
- 只用 `~np.isnan(Z_sparse)` 过滤，未统一过滤 `inf` 或 XYZ 任一维非有限值。
- `project_to_plane_normal(pts_sparse)` 遇到 NaN 会导致 `np.linalg.svd()` 失败。
- 指标统计在空数组或全 NaN 时会触发 `np.nanmax()`、`np.nanmin()` 异常。

建议：

- 用 `np.isfinite(points).all(axis=1)` 统一过滤点。
- 拟合前要求有效点数不少于 3。
- 用 SVD 奇异值检查点集是否退化，例如第二大奇异值过小时判定为共线。
- 对空指标数组返回明确错误，而不是让 NumPy 抛出底层异常。

### P1: 稠密插值点默认参与指标统计，可能放大插值误差

位置：

- `stereo_process.py:72-84`
- `stereo_process.py:93-105`
- `interp.py:5-29`

当前 `densify=True` 时，最终指标默认使用稠密点云距离：

```python
use_d = result["dists_dense"] if result["pts_dense"] is not None else dists_sparse
```

同时 `densify_disparity()` 会用最近邻补全 cubic 插值产生的 NaN，这会把稀疏角点之外的大量区域也填满。对于平整度指标，这可能带来几个风险：

- 稀疏测量点被扩展成大量插值点，指标看起来更稳定，但不一定更真实。
- 最近邻补全会在有效测量区域外制造大片重复视差。
- 全图像素数量远大于真实角点数量，会改变统计权重。

建议：

- 默认用稀疏真实测量点计算核心平整度指标。
- 稠密点云主要用于可视化，或单独输出 `dense_metrics`。
- 若必须用稠密指标，应限制在有效角点凸包、玻璃掩膜或可信 ROI 内。
- 输出 `metric_source` 字段，明确指标来自 `sparse` 还是 `dense`。

### P1: RMS 指标与标准差重复

位置：

- `stereo_process.py:97-105`

当前 `flatness_rms_mm` 计算方式为：

```python
sqrt(mean((d - mean(d)) ** 2))
```

这和 `np.nanstd(use_d_mm)` 本质相同，因此 `flatness_rms_mm` 与 `flatness_std_mm` 是重复指标。

建议二选一：

- 若保留 RMS 名称，则改为 `sqrt(mean(d ** 2))`。
- 若业务想表达去均值后的离散程度，则将字段命名为 `flatness_std_mm`，不要再叫 RMS。

考虑到拟合平面本身已包含 offset，理想情况下残差均值应接近 0，直接 RMS 更符合“相对拟合平面的均方根偏差”。

### P1: 相机标定参数仍为硬编码示例值

位置：

- `detect.py:226-231`
- `pointcloud_gen/main.py` 中相机参数初始化

平面拟合本身无法弥补上游三维重建误差。当前相机内参和 baseline 仍是硬编码值，并且不同入口中的 baseline 不一致。

建议：

- 使用真实相机标定文件加载 `K`、畸变参数、外参和 baseline。
- 在深度恢复前进行双目标定、去畸变和立体校正。
- 在输出结果中记录本次使用的标定版本，便于复现实验。

### P2: 平面拟合 API 表达能力不足

位置：

- `plane_fit.py:3-20`
- `stereo_process.py:54-63`

当前拟合函数只返回 `(a, b, c)` 和 `normal`，缺少内点掩码、残差统计和退化状态等信息。后续若要排查异常样本，不容易定位问题。

建议定义结构化返回值，例如：

```python
@dataclass
class PlaneFitResult:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    coeffs_abc: tuple[float, float, float] | None
    inlier_mask: np.ndarray
    residuals: np.ndarray
    method: str
```

这样可以让指标统计、可视化、导出和前端展示共用同一份模型元数据。

### P2: 缺少针对平面拟合的单元测试

建议增加最小测试集：

- 完美平面点：验证 normal、offset 和距离接近 0。
- 带高斯噪声平面点：验证拟合误差在预期范围内。
- 有异常点：验证鲁棒拟合能剔除异常值。
- 少于 3 个点：验证抛出清晰错误。
- 共线点：验证抛出退化错误。
- 含 NaN/inf 点：验证过滤或报错行为一致。
- 倾斜平面：验证 SVD 正交拟合优于 Z 方向最小二乘。

## 建议落地顺序

1. 在 `plane_fit.py` 中新增通用 SVD 平面拟合函数和通用距离函数。
2. 在 `process_stereo_matches()` 中统一过滤有限点，补充点数和退化检查。
3. 用“初拟合 -> 残差 MAD -> 重拟合”的方式替换原始 Z 值 MAD。
4. 让距离计算、指标统计、投影可视化全部复用同一个 `PlaneFitResult`。
5. 明确距离正负方向，并同步调整后端文案和前端颜色说明。
6. 将核心指标默认切回稀疏真实测量点，稠密指标单独输出。
7. 增加平面拟合单元测试和一组固定样本回归测试。
8. 引入真实相机标定配置，替换硬编码内参和 baseline。

## 推荐目标状态

最终建议形成一个单一可信平面模型：

```text
输入有限点云
  -> SVD 初拟合
  -> 正交残差鲁棒剔除
  -> SVD 重拟合
  -> 统一 PlaneFitResult
  -> 指标统计、投影展示、文件导出、前端颜色全部使用同一残差数组
```

这样可以降低算法解释成本，避免指标和可视化不一致，并提高异常匹配、倾斜姿态和插值误差下的稳定性。
