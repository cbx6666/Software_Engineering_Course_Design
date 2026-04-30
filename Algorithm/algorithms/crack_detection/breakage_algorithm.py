import os
import uuid
import cv2
from .preprocessor import Preprocessor
from .feature_extractor import FeatureExtractor
from .classifier import Classifier
from .config import Config
from typing import Dict

# --------------------------
# 5. 算法主入口
# --------------------------
class GlassBreakageAlgorithm:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.extractor = FeatureExtractor()
        self.classifier = Classifier()
        # 初始化时创建Canny保存文件夹
        os.makedirs(Config.CANNY_SAVE_PATH, exist_ok=True)

    def run(self, image_input: str, method: str = 'traditional') -> Dict:
        """
        算法主流程
        :param image_input: 图像Base64字符串或本地路径
        :param method: 算法模式 ('traditional', 'yolo', 'combined'暂时空缺)
        :return: 符合前端格式的检测结果
        """
        try:
            # 1. 加载图像
            if image_input.startswith("data:image"):
                # 处理Base64格式（后端传入）
                base64_str = image_input.split(",")[1]
                image = self.preprocessor.base64_to_image(base64_str)
                # Base64 上传图片时生成唯一文件名
                img_name = f"{uuid.uuid4().hex}.png"
            else:
                # 处理本地路径（测试用）
                image = cv2.imread(image_input)
                img_name = os.path.basename(image_input)

            # 2. 预处理
            processed_img = self.preprocessor.preprocess(image)

            # 3. 特征提取
            features = {}
            
            # --- 传统算法提取 ---
            edges, crack_ratio = self.extractor.extract_edge(processed_img)
            is_break = self.extractor.is_glass_break(edges)
            glcm_features = self.extractor.extract_glcm_features(processed_img)
            
            # 保存 Canny 边缘图到本地
            canny_save_path = os.path.join(Config.CANNY_SAVE_PATH, img_name)
            cv2.imwrite(canny_save_path, edges)  
            if Config.IS_PRINT: print(f"Canny边缘图像已保存至：{canny_save_path}")

            # --- YOLO 特征提取 (解开注释并传入原图) ---
            yolo_detections = []
            if method in ['yolo', 'combined']:
                yolo_detections = self.extractor.extract_yolo_features(image)

            # 4. 整合特征
            features = {
                "crack_ratio": crack_ratio,
                "glcm": glcm_features,
                "is_break": is_break,
                "yolo": yolo_detections
            }

            # 5. 分类决策
            status, description, damage_area = self.classifier.predict_risk(features, method=method)

            # 6. 封装结果
            return {
                "status": status,
                "title": {
                    "success": "检测正常",
                    "warning": "自爆风险预警",
                    "error": "自爆确认"
                }[status],
                "description": description,
                "details": [
                    {"label": "损伤面积", "value": f"{damage_area:.2f} mm²"} 
                    if damage_area > 0 else {"label": "检查项", "value": "无明显物理损伤"}
                ]
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "title": "系统错误",
                "description": f"算法执行失败: {str(e)}",
                "details": []
            }