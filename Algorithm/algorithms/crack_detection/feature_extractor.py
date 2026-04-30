import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from typing import Tuple, Dict, List
from .config import Config
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# --------------------------
# 3. 特征提取模块
# --------------------------
class FeatureExtractor:
    
    # 类级变量，用于缓存 YOLO 模型，避免每次检测都重新加载导致卡顿
    _yolo_model = None 

    @staticmethod
    def extract_edge(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """边缘检测（CANNY算子）"""
        edges = cv2.Canny(image, 100, 240)
        # 计算裂纹面积占比（像素数/总像素）
        total_pixels = image.size
        crack_pixels = np.sum(edges > 0)
        crack_ratio = crack_pixels / total_pixels
        return edges, crack_ratio

    @staticmethod
    def extract_glcm_features(image: np.ndarray) -> Dict[str, float]:
        """提取GLCM特征（能量、熵、相关性，赵珂2022）"""
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
            "energy": np.mean(graycoprops(glcm, 'energy')),  # 能量
            "entropy": shannon_entropy(image),               # 改用图像直接计算熵，兼容所有版本
            "correlation": np.mean(graycoprops(glcm, 'correlation'))  # 相关性
        }
    
    @staticmethod
    def is_glass_break(edge_image: np.ndarray) -> bool:
        """
        通过轮廓分析判断边缘图像中是否包含玻璃自爆特征
        """
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if Config.IS_PRINT:print("未检测到任何轮廓")
            return False
        
        if Config.IS_PRINT:print(f"共检测到 {len(contours)} 个轮廓，开始分析特征参数...")
        contour_idx = 0 
        
        for contour in contours:
            contour_idx += 1
            area = cv2.contourArea(contour)
            
            if area < Config.MIN_CONTOUR_AREA:  
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            
            # === 1. 核心几何特征 ===
            shape_factor = (4 * np.pi * area) / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            
            # === 2. [抗网格/直电线补丁]：凸包矩形度 ===
            hull_rect = cv2.minAreaRect(hull) 
            hull_rect_area = hull_rect[1][0] * hull_rect[1][1]
            hull_boxiness = float(hull_area) / hull_rect_area if hull_rect_area != 0 else 0
            
            # === 3. [L型/U型大色块补丁]：真实厚度 + 顶点数 ===
            real_thickness = min(hull_rect[1][0], hull_rect[1][1]) 
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertex_count = len(approx)
            
            if Config.IS_PRINT:
                print(f"\n轮廓 {contour_idx} 特征参数：")
                print(f"  形状因子：{shape_factor:.4f}（阈值 < {Config.SHAPE_FACTOR_THRESHOLD}）")
                print(f"  宽高比：{aspect_ratio:.4f}（阈值 {Config.ASPECT_RATIO_LOW}~{Config.ASPECT_RATIO_HIGH}）")
                print(f"  密实度：{solidity:.4f}（阈值 {Config.SOLIDITY_LOW}~{Config.SOLIDITY_HIGH}）")
                print(f"  凸包矩形度：{hull_boxiness:.4f} (超 {Config.BOXINESS_THRESHOLD} 为规则网格)")
                print(f"  真实厚度: {real_thickness:.2f} | 顶点数: {vertex_count} (厚度>{Config.L_FRAME_THICKNESS_MIN}且顶点<{Config.L_FRAME_VERTEX_MAX}为L型窗框)")
            
            # === 条件判断 ===
            if (shape_factor < Config.SHAPE_FACTOR_THRESHOLD and 
                Config.ASPECT_RATIO_LOW < aspect_ratio < Config.ASPECT_RATIO_HIGH and 
                Config.SOLIDITY_LOW < solidity < Config.SOLIDITY_HIGH):
                
                # 凸包矩形度判断
                if hull_boxiness > Config.BOXINESS_THRESHOLD:
                    continue
                
                # L型/U型 窗框拐角判断
                if real_thickness > Config.L_FRAME_THICKNESS_MIN and vertex_count < Config.L_FRAME_VERTEX_MAX:
                    continue
                
                if Config.IS_PRINT: print(f" 🚨 轮廓 {contour_idx}：符合裂纹特征")
                return True
                
            else:
                if Config.IS_PRINT: print(f" 🚫 轮廓 {contour_idx}：不符合自爆特征")
        
        if Config.IS_PRINT:print("\n✅ 所有轮廓均不符合自爆特征")
        return False
    
    @staticmethod
    def extract_yolo_features(image: np.ndarray) -> List[Dict]:
        """
        用YOLO 模型检测玻璃面板异常区域
        输入应为原始彩色 numpy 数组
        """
        if YOLO is None:
            if Config.IS_PRINT: print("未检测到 ultralytics 库，无法运行 YOLO。")
            return []

        # 懒加载模式：只有第一次调用时才加载模型到显存
        if FeatureExtractor._yolo_model is None:
            if Config.IS_PRINT: print("正在首次加载 YOLO 模型至显存...")
            FeatureExtractor._yolo_model = YOLO(Config.YOLO_MODEL_PATH)
            
        # 进行推理
        results = FeatureExtractor._yolo_model(image, verbose=False) 
        
        detections = []
        for result in results:
            # 使用的是 yolov8n.pt 跑出的标准矩形框模型，使用 result.boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > Config.YOLO_CONF_THRESHOLD:
                        detections.append({
                            "coords": box.xyxy[0].tolist(),  # 标准框坐标
                            "confidence": conf,
                            "class": int(box.cls)            # 类别
                        })
        return detections