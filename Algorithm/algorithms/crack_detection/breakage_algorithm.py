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
        #初始化时创建Canny保存文件夹
        os.makedirs(Config.CANNY_SAVE_PATH, exist_ok=True)

    def run(self, image_input: str) -> Dict:
        """
        算法主流程
        :param image_input: 图像Base64字符串或本地路径
        :param glass_id: 玻璃编号（可选，用于定位具体幕墙）
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
                # 本地图像命名：取原文件名
                img_name = os.path.basename(image_input)
            
            # 检查图像是否加载成功
            if image is None:
                raise ValueError(f"无法加载图像，请检查路径：{image_input} 或Base64编码是否有效")

            # 2. 预处理
            processed_img = self.preprocessor.preprocess(image)

            # 3. 特征提取
            edges, crack_ratio = self.extractor.extract_edge(processed_img)

            # 关键改动1：保存Canny边缘图像（灰度图→彩色图，便于查看）
            
            canny_save_path = os.path.join(Config.CANNY_SAVE_PATH, img_name)
            cv2.imwrite(canny_save_path, edges)  # 保存图像到output文件夹
            print(f"Canny边缘图像已保存至：{canny_save_path}")

            # 关键改动2：实时显示Canny边缘图像（可选，运行时会弹出窗口）
            cv2.imshow("Canny Edge Detection", edges)
            cv2.waitKey(0)  # 按任意键关闭窗口
            cv2.destroyAllWindows()  # 释放窗口资源


            glcm_features = self.extractor.extract_glcm_features(processed_img)
            # 注释YOLO特征提取（暂不使用）
            # yolo_detections = self.extractor.extract_yolo_features(processed_img)  # 刘长儒2024

            # 4. 整合特征（移除YOLO相关）
            features = {
                "crack_ratio": crack_ratio,
                "glcm": glcm_features
                # 注释YOLO特征（暂不使用）
                # "yolo": yolo_detections
            }

            # 5. 分类决策
            status, description, damage_area = self.classifier.predict_risk(features)

            # 6. 封装结果（匹配前端接口格式）
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
                    if damage_area > 0 else None
                ],
            }

        except Exception as e:
            # 异常处理：返回错误信息
            return {
                "status": "error",
                "title": "算法执行失败",
                "description": f"处理过程出错：{str(e)}",
                "details": None
            }