import numpy as np

# --------------------------
# 1. 配置与常量定义
# --------------------------
class Config:
    """算法配置参数"""
    # 图像预处理参数
    RESIZE_SIZE = (640, 480)  # 统一图像尺寸
    MAX_IMAGE_SIZE = 640   # 统一图像最长边大小
    MEDIAN_FILTER_KERNEL = 3  # 中值滤波核大小
    # 特征提取参数
    GLCM_DISTANCES = [1]  # 灰度共生矩阵距离
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 角度
    # 分类阈值
    CRACK_AREA_THRESHOLD = 50  # 裂纹面积阈值（mm²），超过则判定为自爆
    WARNING_THRESHOLD = 10     # 警告阈值（小裂纹）

    # 轮廓分析阈值
    MIN_CONTOUR_AREA = 100  # 最小轮廓面积（过滤噪声）
    SHAPE_FACTOR_THRESHOLD = 0.3  # 形状因子阈值
    ASPECT_RATIO_LOW = 0.3  # 宽高比下限
    ASPECT_RATIO_HIGH = 3   # 宽高比上限
    SOLIDITY_LOW = 0.04      # 密实度下限
    SOLIDITY_HIGH = 0.8     # 密实度上限
    MAX_PAIR_DISTANCE = 100 # 两个疑似碎片的中心点距离小于此值，才判定为一对自爆碎片

    BOXINESS_THRESHOLD = 0.88     # 凸包矩形度上限
    L_FRAME_THICKNESS_MIN = 15.0  # L型窗框的最小真实厚度 (像素)
    L_FRAME_VERTEX_MAX = 22       # L型窗框的最大顶点数 (超过则认为边缘足够复杂，可能是裂纹)

    # YOLO模型路径
    YOLO_MODEL_PATH = "./models/yolo_crack.pt"  # 确保路径指向你的 pt 文件
    YOLO_CONF_THRESHOLD = 0.3  # 置信度阈值：YOLO 只有在确信度 > 50% 时才报警

    # 模型路径
    SVM_MODEL_PATH = "./models/svm_model.pkl"  # SVM模型（暂未使用）

    ### TEST
    # Canny边缘图像保存路径
    CANNY_SAVE_PATH = "./canny_edges/"  # 边缘图像输出文件夹
    # 批量测试配置
    TEST_IMAGE_DIR = "./test_images/"  # 测试图像文件夹路径
    # 是否打印当前轮廓的详细参数
    IS_PRINT = False

    # 是否移动错误样本（功能开关）
    # True: 开启移动，原文件会被移走
    # False: 关闭此功能，只统计不移动
    IS_MOVE_ERROR_SAMPLES = False

    # 误报/漏报样本保存路径
    REMOVE_IMAGE_DIR = "./datasets/images_remove"  # 保存错误的图片
    REMOVE_LABEL_DIR = "./datasets/labels_remove"  # 保存对应的标签