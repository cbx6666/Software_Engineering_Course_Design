# 平整度检测前 3 流程优化说明

## 1. 优化边界

本次优化只覆盖平整度检测前 3 个流程：

1. 投影反射差分
2. 棋盘格角点检测
3. 左右角点匹配

保持兼容的内容：

- 四图输入仍为 `left_env`、`left_mix`、`right_env`、`right_mix`
- 主入口仍为 `algorithms.flatness_detection.detect.main(data_dir, result_dir)`
- 原有输出仍保留：`left_detect.png`、`right_detect.png`、`left_mask.png`、`right_mask.png`、`corners_left.json`、`corners_right.json`
- FastAPI 服务层返回结构未修改
- 点云生成、平面拟合、平整度指标计算未作为本次重点修改

## 2. 新增配置与日志

新增统一配置模块：

- `Algorithm/algorithms/flatness_detection/config.py`

主要配置类：

- `FlatnessConfig`
- `ProjectionDiffConfig`
- `CornerDetectionConfig`
- `CornerMatchingConfig`
- `LoggingConfig`
- `DebugOutputConfig`

新增轻量日志与错误类型：

- `Algorithm/algorithms/flatness_detection/logging_utils.py`
- `Algorithm/algorithms/flatness_detection/errors.py`

默认日志级别为 `INFO`。Debug 输出默认关闭；开启后可输出对齐图、差分图、mask、角点图和匹配可视化图。

## 3. 投影反射差分优化

涉及文件：

- `projector_reflection_diff/core/alignment.py`
- `projector_reflection_diff/core/brightness_matching.py`
- `projector_reflection_diff/core/chessboard_processing.py`
- `projector_reflection_diff/main.py`

优化点：

- ECC 对齐增加 fallback 链：`homography -> affine -> translation -> identity`
- ECC 输入统一灰度化、归一化、降噪，并捕获 `cv2.error`
- 支持下采样估计变换，再映射回原图，提高大图对齐速度
- 亮度匹配 mask 会排除明显变化区域、过亮区域和过暗区域
- 亮度线性拟合改为鲁棒闭式拟合，避免构造大矩阵
- 有效样本过少时退化为中位数偏移补偿
- 棋盘格 mask 增加 open/close、填洞、小连通域删除、保留最大区域等后处理
- 中间调试图只在 debug 开启时输出，默认不污染 `result_dir`

## 4. 棋盘格角点检测优化

涉及文件：

- `stereo_corner_matching/core/corner_detection.py`
- `stereo_corner_matching/core/image_processing.py`

优化点：

- 角点检测封装为多策略流程
- 优先使用 `findChessboardCornersSB`
- fallback 到 `findChessboardCorners`
- 失败后依次尝试 `original`、`CLAHE`、`blur`、`threshold`、`sharpen`
- 兼容不同 OpenCV 返回格式
- 兼容 `CALIB_CB_PARTIAL_OK` 不存在的版本
- 增加角点质量校验：
  - 数量是否足够
  - 覆盖区域是否过小
  - 是否存在重复点
  - 平均间距是否异常
  - 完整棋盘格排序是否单调
- `cornerSubPix` 前统一图像类型和角点格式
- `cornerSubPix` 失败时保留原角点并写 warning，不让流程直接崩溃

## 5. 左右角点匹配优化

涉及文件：

- `stereo_corner_matching/core/corner_matching.py`
- `detect.py`

优化点：

- 使用角点整体几何结构估计棋盘格行列方向
- 将角点投影到估计的行列轴上，再聚类到固定网格
- 缺失点用 `NaN` 占位，避免简单按 y 分组造成错位
- 匹配后执行几何一致性校验：
  - 视差是否为正
  - 视差是否存在离群点
  - 左右点 y 坐标差是否异常
  - 异常点比例是否过高
- 可剔除少量异常匹配；异常比例过高或剩余点过少时报错
- 新增兼容性输出 `match_quality.json`

`match_quality.json` 包含：

- `matched_count`
- `expected_count`
- `match_ratio`
- `disparity_mean`
- `disparity_std`
- `outlier_count`

## 6. API 兼容性

保持兼容：

- `detect.main(data_dir, result_dir)` 仍可直接调用
- 新增 `config` 参数是可选参数，不影响旧调用
- `process_projection_diff`、`detect_corners` 等内部函数也保持默认配置
- 最终角点 JSON 仍是二维点数组：`[[x1, y1], [x2, y2], ...]`
- FastAPI 服务层未修改

新增输出：

- `match_quality.json`

该文件为额外质量指标，不影响旧流程读取 `corners_left.json` 和 `corners_right.json`。

## 7. 新增测试

新增测试文件：

- `Algorithm/tests/test_flatness_front_flows.py`

覆盖内容：

- 投影差分：
  - env/mix 尺寸不同
  - ECC 失败 fallback
  - 明显亮度变化
  - 棋盘格 mask 小连通域过滤
- 角点检测：
  - 正常棋盘格
  - 低对比度棋盘格
  - `cornerSubPix` 异常 fallback
  - OpenCV 返回格式兼容路径
- 左右角点匹配：
  - 完整角点网格
  - 缺失部分角点
  - 视差异常点剔除
  - 匹配点过少时报错
- 回归兼容：
  - `detect.main(data_dir, result_dir)` 可被旧服务调用
  - 输出 `corners_left.json`、`corners_right.json`、`flatness_metrics.json`
  - JSON 格式保持兼容

运行方式：

```bash
python -m unittest discover -s Algorithm/tests -v
```

编译检查：

```bash
python -m compileall -q Algorithm
```

## 8. 仍需后续真实标定解决的问题

本次未重点修改第 4 步点云生成和平整度计算，因此以下问题仍需要后续结合真实标定处理：

- 当前相机内参 `K` 和 `baseline` 仍是示例值
- 真实场景中需使用实际双目标定参数
- 强透视、强反光、棋盘严重遮挡时，仍建议结合真实数据调参
- 平整度数值准确性最终取决于标定质量、角点物理对应关系和后续点云重建模型
