import os
import uuid
from typing import Dict

import cv2

from .classifier import Classifier
from .config import Config
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor


class GlassBreakageAlgorithm:
    """玻璃裂纹/自爆检测主入口。"""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.extractor = FeatureExtractor()
        self.classifier = Classifier()

        if Config.SAVE_CANNY_IMAGE:
            os.makedirs(Config.CANNY_SAVE_PATH, exist_ok=True)

    def run(self, image_input: str, method: str = "traditional") -> Dict:
        """执行裂纹/自爆检测流程。"""
        try:
            if image_input.startswith("data:image"):
                base64_str = image_input.split(",")[1]
                image = self.preprocessor.base64_to_image(base64_str)
                img_name = f"{uuid.uuid4().hex}.png"
            else:
                image = cv2.imread(image_input)
                img_name = os.path.basename(image_input)

            processed_img = self.preprocessor.preprocess(image)

            edges, crack_ratio = self.extractor.extract_edge(processed_img)
            is_break = self.extractor.is_glass_break(edges)
            glcm_features = self.extractor.extract_glcm_features(processed_img)

            if Config.SAVE_CANNY_IMAGE:
                canny_save_path = os.path.join(Config.CANNY_SAVE_PATH, img_name)
                cv2.imwrite(canny_save_path, edges)
                if Config.IS_PRINT:
                    print(f"Canny边缘图像已保存至：{canny_save_path}")

            yolo_detections = []
            if method in ["yolo", "combined"]:
                yolo_detections = self.extractor.extract_yolo_features(image)

            features = {
                "crack_ratio": crack_ratio,
                "glcm": glcm_features,
                "is_break": is_break,
                "yolo": yolo_detections,
            }

            status, description, damage_area = self.classifier.predict_risk(features, method=method)

            return {
                "status": status,
                "title": {
                    "success": "检测正常",
                    "warning": "自爆风险预警",
                    "error": "自爆确认",
                }[status],
                "description": description,
                "details": [
                    {"label": "损伤面积", "value": f"{damage_area:.2f} mm²"}
                    if damage_area > 0
                    else {"label": "检查项", "value": "无明显物理损伤"}
                ],
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "title": "系统错误",
                "description": f"算法执行失败: {str(e)}",
                "details": [],
            }
