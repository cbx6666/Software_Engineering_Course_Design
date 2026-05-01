import numpy as np


class Config:
    """裂纹/自爆检测算法配置。"""

    # 图像预处理参数
    RESIZE_SIZE = (640, 480)
    MAX_IMAGE_SIZE = 640
    MEDIAN_FILTER_KERNEL = 3

    # 纹理特征参数
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # 分类阈值
    CRACK_AREA_THRESHOLD = 50
    WARNING_THRESHOLD = 10

    # 轮廓分析阈值
    MIN_CONTOUR_AREA = 100
    SHAPE_FACTOR_THRESHOLD = 0.3
    ASPECT_RATIO_LOW = 0.3
    ASPECT_RATIO_HIGH = 3
    SOLIDITY_LOW = 0.04
    SOLIDITY_HIGH = 0.8
    MAX_PAIR_DISTANCE = 100

    BOXINESS_THRESHOLD = 0.88
    L_FRAME_THICKNESS_MIN = 15.0
    L_FRAME_VERTEX_MAX = 22

    # YOLO 模型路径
    YOLO_MODEL_PATH = "./models/yolo_crack.pt"
    YOLO_CONF_THRESHOLD = 0.3

    # 传统模型路径
    SVM_MODEL_PATH = "./models/svm_model.pkl"

    # 调试与测试相关配置
    SAVE_CANNY_IMAGE = False  # 是否保存 Canny 边缘图，默认关闭
    CANNY_SAVE_PATH = "./canny_edges/"
    TEST_IMAGE_DIR = "./test_images/"
    IS_PRINT = False

    # 是否移动误报/漏报样本
    IS_MOVE_ERROR_SAMPLES = False
    REMOVE_IMAGE_DIR = "./datasets/images_remove"
    REMOVE_LABEL_DIR = "./datasets/labels_remove"
