import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from typing import Tuple, Dict
from .config import Config

# --------------------------
# 3. 特征提取模块
# --------------------------
class FeatureExtractor:
    @staticmethod
    def extract_edge(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """边缘检测（SUSAN算子+区域分裂合并，赵珂2022）"""
        # 简化实现：用Canny边缘检测替代SUSAN（实际需复现文献算法）
        edges = cv2.Canny(image, 100, 240)
        # 计算裂纹面积占比（像素数/总像素）
        total_pixels = image.size
        crack_pixels = np.sum(edges > 0)
        crack_ratio = crack_pixels / total_pixels
        return edges, crack_ratio

    @staticmethod
    def extract_glcm_features(image: np.ndarray) -> Dict[str, float]:
        """提取GLCM特征（能量、熵、相关性，赵珂2022）"""
        from skimage.feature import graycomatrix, graycoprops
        from skimage.measure import shannon_entropy  # 新增导入
        # 确保图像是8位灰度图
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # 计算灰度共生矩阵
        glcm = graycomatrix(
            image,
            distances=Config.GLCM_DISTANCES,
            angles=Config.GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True
        )
        # 提取特征
        return {
            "energy": np.mean(graycoprops(glcm, 'energy')),#能量
            "entropy": shannon_entropy(image),  # 改用图像直接计算熵，兼容所有版本
            "correlation": np.mean(graycoprops(glcm, 'correlation'))#相关性
        }

    # 注释YOLO特征提取方法（暂不使用）
    # @staticmethod
    # def extract_yolo_features(image: np.ndarray) -> List[Dict]:
    #     """用YOLOv8-obb检测玻璃面板及异常区域（刘长儒2024）"""
    #     model = YOLO(Config.YOLO_MODEL_PATH)
    #     results = model(image, verbose=False)  # 不打印日志
    #     # 解析结果：提取每个检测框的坐标和置信度
    #     detections = []
    #     for result in results:
    #         for box in result.obb:  # 旋转框（obb）
    #             detections.append({
    #                 "coords": box.xyxyxyxy.tolist(),  # 旋转框坐标
    #                 "confidence": float(box.conf),    # 置信度
    #                 "class": int(box.cls)             # 类别（0:正常, 1:裂纹, 2:自爆）
    #             })
    #     return detections
