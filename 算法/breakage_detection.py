"""
玻璃幕墙自爆检测算法层
输入：图像数据（Base64/路径）+ 配置参数
输出：符合前端格式的检测结果（status/description等）
技术参考：赵珂2022（传统视觉）、刘长儒2024（YOLO+多源）
"""
import os
import base64
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch
# 注释YOLO相关导入（暂不使用）
# from ultralytics import YOLO  # YOLOv8库（刘长儒2024文献用到）
from typing import Dict, List, Optional, Tuple


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
    # 注释YOLO模型路径（暂不使用）
    # YOLO_MODEL_PATH = "./models/yolov8_obb.pt"  # YOLOv8-obb模型（刘长儒2024）


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


# --------------------------
# 4. 分类与决策模块
# --------------------------
class Classifier:
    def __init__(self):
        # 加载预训练模型（SVM，赵珂2022）
        self.svm = self._load_svm_model()
        self.scaler = StandardScaler()  # 特征标准化

    def _load_svm_model(self) -> SVC:
        """加载SVM模型（实际需训练后保存）"""
        if os.path.exists(Config.SVM_MODEL_PATH):
            import joblib
            return joblib.load(Config.SVM_MODEL_PATH)
        else:
            # 若模型不存在，初始化一个RBF核SVM（后续需用样本训练）
            return SVC(kernel='rbf', probability=True)

    #def predict_risk(self, features: Dict) -> Tuple[str, str, float]:
    #    """
    #    风险预测：结合特征判定状态
    #    返回：status（success/warning/error）、描述、损伤面积（mm²）
    #    """
    #    # 1. 计算实际损伤面积（假设玻璃实际尺寸为1000x1000mm）
    #    crack_ratio = features.get("crack_ratio", 0)
    #    damage_area = crack_ratio * 1000 * 1000  # 像素占比→实际面积

        # 2. 基于SVM预测（模拟，实际需用特征向量输入）
        # 特征向量：[energy, entropy, correlation, crack_ratio]
    #    feature_vec = [
    #        features["glcm"]["energy"],
    #        features["glcm"]["entropy"],
    #        features["glcm"]["correlation"],
    #        crack_ratio
    #    ]
    #    feature_vec = self.scaler.fit_transform([feature_vec])
    #    pred = self.svm.predict(feature_vec)[0]

        # 3. 结合阈值判定结果
    #    if damage_area > Config.CRACK_AREA_THRESHOLD or pred == 2:
    #        return "error", "检测到玻璃自爆，损伤面积较大", damage_area
    #    elif damage_area > Config.WARNING_THRESHOLD or pred == 1:
    #        return "warning", "玻璃存在裂纹，有自爆风险", damage_area
    #    else:
    #        return "success", "玻璃状态正常，无自爆风险", 0.0
        
    def predict_risk(self, features: Dict) -> Tuple[str, str, float]:
        """风险预测：暂时跳过SVM，仅用面积阈值判断"""
        # 1. 计算实际损伤面积
        crack_ratio = features.get("crack_ratio", 0)
        damage_area = crack_ratio * 1000 * 1000  # 像素占比→实际面积

        # 2. 仅用面积阈值判断（跳过SVM预测）
        if damage_area > Config.CRACK_AREA_THRESHOLD:
            return "error", "检测到玻璃自爆，损伤面积较大", damage_area
        elif damage_area > Config.WARNING_THRESHOLD:
            return "warning", "玻璃存在裂纹，有自爆风险", damage_area
        else:
            return "success", "玻璃状态正常，无自爆风险", 0.0


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

    def run(self, image_input: str, glass_id: str = "") -> Dict:
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
                "glass_id": glass_id,
                "timestamp": self._get_timestamp()
            }

        except Exception as e:
            # 异常处理：返回错误信息
            return {
                "status": "error",
                "title": "算法执行失败",
                "description": f"处理过程出错：{str(e)}",
                "details": None
            }

    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳（用于日志）"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --------------------------
# 批量测试入口
# --------------------------
def batch_test():
    """批量测试test_images文件夹，计算正确率"""
    algorithm = GlassBreakageAlgorithm()
    total = 0  # 参与计算的样本总数
    correct = 0  # 检测正确的样本数
    failed = 0   # 检测失败的样本数
    skipped = 0  # 跳过的样本数（文件名不匹配规则）

    print("="*60)
    print(f"批量测试开始：{Config.TEST_IMAGE_DIR}")
    print("规则：文件名含'normal'→正常；含'crack'→异常")
    print("="*60)

    for filename in os.listdir(Config.TEST_IMAGE_DIR):
        # 过滤非图像文件
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            skipped += 1
            print(f"⏭️  跳过非图像：{filename}")
            continue

        image_path = os.path.join(Config.TEST_IMAGE_DIR, filename)
        print(f"\n📌 检测图像：{filename}")
        result = algorithm.run(image_path, glass_id=f"测试-{filename}")

        # 处理检测失败的情况
        if result["status"] == "error" and "处理出错" in result["description"]:
            failed += 1
            print(f"❌ 检测失败：{result['description']}")
            continue

        # 判定真实标签与预测结果
        true_label = None
        if "normal" in filename.lower():
            true_label = "normal"
        elif "crack" in filename.lower():
            true_label = "abnormal"
        else:
            skipped += 1
            print(f"⏭️  文件名不匹配规则，跳过：{filename}")
            continue

        # 预测结果转换（success→normal；warning/error→abnormal）
        pred_label = "normal" if result["status"] == "success" else "abnormal"

        # 统计正确数
        total += 1
        if true_label == pred_label:
            correct += 1
            print(f"✅ 检测正确：真实[{true_label}]，预测[{pred_label}]")
        else:
            print(f"❌ 检测错误：真实[{true_label}]，预测[{pred_label}]")

    # 计算正确率（处理除数为0的情况）
    accuracy = (correct / total) * 100 if total > 0 else 0

    # 输出最终统计结果
    print("\n" + "="*60)
    print("批量测试结果汇总：")
    print(f"总测试样本数：{total}")
    print(f"正确数：{correct}，错误数：{total - correct}")
    print(f"检测失败数：{failed}，跳过数：{skipped}")
    print(f"正确率：{accuracy:.2f}%")
    print("="*60)


# --------------------------
# 主函数入口
# --------------------------
if __name__ == "__main__":
    # 执行批量测试并输出正确率
    batch_test()