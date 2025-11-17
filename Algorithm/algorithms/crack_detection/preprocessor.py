import cv2
import numpy as np
import base64
from .config import Config

# --------------------------
# 2. 数据预处理模块
# --------------------------
class Preprocessor:
    @staticmethod
    def base64_to_image(base64_str: str) -> np.ndarray:
        """将Base64编码转换为OpenCV图像（后端传入的图像格式）"""
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """图像预处理： resize + 降噪 + 灰度化（赵珂2022文献步骤）"""
        # 1. 统一尺寸
        resized = cv2.resize(image, Config.RESIZE_SIZE)
        # 2. 中值滤波降噪（优化中值滤波）
        filtered = cv2.medianBlur(resized, Config.MEDIAN_FILTER_KERNEL)
        # 3. 加权灰度化（文献：加权灰度化）
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        # 4. 灰度拉伸（增强对比度）
        stretched = Preprocessor.gray_stretch(gray)
        return stretched

    @staticmethod
    def gray_stretch(gray_img: np.ndarray) -> np.ndarray:
        """灰度拉伸（优化版，赵珂2022）"""
        min_val = np.min(gray_img)
        max_val = np.max(gray_img)
        if max_val == min_val:
            return np.zeros_like(gray_img)
        return ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
