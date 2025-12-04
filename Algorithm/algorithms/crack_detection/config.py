import numpy as np

# --------------------------
# 1. 配置与常量定义
# --------------------------
class Config:
    """算法配置参数"""
    # 图像预处理参数
    RESIZE_SIZE = (640, 480)  # 统一图像尺寸
    MEDIAN_FILTER_KERNEL = 3  # 中值滤波核大小（赵珂2022）
    # 特征提取参数
    GLCM_DISTANCES = [1]  # 灰度共生矩阵距离（赵珂2022）
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 角度
    # 分类阈值
    CRACK_AREA_THRESHOLD = 50  # 裂纹面积阈值（mm²），超过则判定为自爆
    WARNING_THRESHOLD = 10     # 警告阈值（小裂纹）
    # 模型路径
    SVM_MODEL_PATH = "./models/svm_model.pkl"  # SVM模型（赵珂2022）
    # 新增：Canny边缘图像保存路径
    CANNY_SAVE_PATH = "./canny_edges/"  # 边缘图像输出文件夹
    # 批量测试配置
    TEST_IMAGE_DIR = "./test_images/"  # 测试图像文件夹路径

    # 新增轮廓分析阈值
    MIN_CONTOUR_AREA = 100  # 最小轮廓面积（过滤噪声）
    SHAPE_FACTOR_THRESHOLD = 0.3  # 形状因子阈值
    ASPECT_RATIO_LOW = 0.3  # 宽高比下限
    ASPECT_RATIO_HIGH = 3   # 宽高比上限
    SOLIDITY_LOW = 0.2      # 密实度下限
    SOLIDITY_HIGH = 0.8     # 密实度上限

    # 注释YOLO模型路径（暂不使用）
    # YOLO_MODEL_PATH = "./models/yolov8_obb.pt"  # YOLOv8-obb模型（刘长儒2024）

