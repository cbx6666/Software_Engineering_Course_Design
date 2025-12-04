import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from typing import Tuple, Dict
from .config import Config
import matplotlib.pyplot as plt

# --------------------------
# 3. 特征提取模块
# --------------------------
class FeatureExtractor:
    @staticmethod
    def extract_edge(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """边缘检测（CANNY算子+区域分裂合并，赵珂2022）"""
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
    
    @staticmethod
    def is_glass_break(edge_image: np.ndarray) -> bool:
        """
        通过轮廓分析判断边缘图像中是否包含玻璃自爆特征
        :param edge_image: Canny边缘检测后的灰度图（0-255）
        :return: True表示可能是自爆，False表示不是
        """
        # 提取轮廓
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未检测到任何轮廓")
            return False
        
        print(f"共检测到 {len(contours)} 个轮廓，开始分析特征参数...")
        contour_idx = 0  # 轮廓索引，用于终端显示
        
        for contour in contours:
            contour_idx += 1
            area = cv2.contourArea(contour)
            
            # 过滤小面积轮廓并打印信息
            if area < Config.MIN_CONTOUR_AREA:  # 建议在Config中定义该阈值
                print(f"轮廓 {contour_idx}：面积 {area:.2f}（小于阈值 {Config.MIN_CONTOUR_AREA}，跳过）")
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                print(f"轮廓 {contour_idx}：周长为0（无效轮廓，跳过）")
                continue
            
            # 计算特征参数
            shape_factor = (4 * np.pi * area) / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            
            # 打印当前轮廓的详细参数
            print(f"\n轮廓 {contour_idx} 特征参数：")
            print(f"  面积：{area:.2f}")
            print(f"  周长：{perimeter:.2f}")
            print(f"  形状因子：{shape_factor:.4f}（阈值 < {Config.SHAPE_FACTOR_THRESHOLD}）")
            print(f"  宽高比：{aspect_ratio:.4f}（阈值范围 {Config.ASPECT_RATIO_LOW}~{Config.ASPECT_RATIO_HIGH}）")
            print(f"  密实度：{solidity:.4f}（阈值范围 {Config.SOLIDITY_LOW}~{Config.SOLIDITY_HIGH}）")
            
            # 判断是否符合自爆特征
            if (shape_factor < Config.SHAPE_FACTOR_THRESHOLD and 
                Config.ASPECT_RATIO_LOW < aspect_ratio < Config.ASPECT_RATIO_HIGH and 
                Config.SOLIDITY_LOW < solidity < Config.SOLIDITY_HIGH):
                print(f"轮廓 {contour_idx}：符合自爆特征！")
                return True
            else:
                print(f"轮廓 {contour_idx}：不符合自爆特征")
        
        print("\n所有轮廓均不符合自爆特征")
        return False

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
