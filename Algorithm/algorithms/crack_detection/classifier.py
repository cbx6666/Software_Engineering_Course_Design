import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
from .config import Config

# --------------------------
# 4. 分类与决策模块
# --------------------------
class Classifier:
    def __init__(self):
        # 加载预训练模型
        # self.svm = self._load_svm_model()
        self.scaler = StandardScaler()  # 特征标准化
        self._svm_instance = None  # 用一个私有变量做缓存

    @property
    def svm(self) -> SVC: # 暂未使用
        """
        惰性加载 SVM 模型 (Lazy Load)
        只有在代码第一次通过 self.svm 访问时，才会执行加载逻辑。
        """
        if self._svm_instance is None:
            self._svm_instance = self._load_svm_model()
        return self._svm_instance

    def _load_svm_model(self) -> SVC: # 暂未使用
        """加载SVM模型（实际需训练后保存）"""
        if os.path.exists(Config.SVM_MODEL_PATH):
            import joblib
            return joblib.load(Config.SVM_MODEL_PATH)
        else:
            # 若模型不存在，初始化一个RBF核SVM
            return SVC(kernel='rbf', probability=True)

    def predict_risk(self, features: Dict, method: str = 'traditional') -> Tuple[str, str, float]:
        """
        根据提取的特征进行风险预测
        :param features: 包含所有特征的字典 (is_break, yolo 等)
        :param method: 期望启用的算法通道 ('traditional' 或 'yolo'，'combined'暂时空缺)
        :return: (status, description, damage_area)
        """
        
        status = "success"
        description = "正常"
        damage_area = 0.0

        # ==========================================
        # 通道 1：纯粹的传统算法逻辑
        # ==========================================
        if method == 'traditional':
            # 这里的特征必须与 extractor.py 中 is_glass_break 的返回值对应
            is_break = features.get("is_break", False)
            
            if is_break:
                status = "error"
                description = "[传统 模式]检测到玻璃裂纹特征"
                damage_area = 100.0  # 示意值，如果你有真实的面积计算可以替换
            else:
                status = "success"
                description = "[传统 模式]未发现明显裂纹"

        # ==========================================
        # 通道 2：纯粹的 YOLO 深度学习逻辑
        # ==========================================
        elif method == 'yolo':
            yolo_detections = features.get("yolo", [])
            
            if len(yolo_detections) > 0:
                status = "error"
                # 提取最高置信度
                best_conf = max([d['confidence'] for d in yolo_detections])
                description = f"[YOLO 模式]检测到玻璃裂纹特征 (置信度: {best_conf:.2f})"
                
                # 简单估算损伤面积 (把所有框的面积加起来作为示意)
                for det in yolo_detections:
                    box = det["coords"]
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    damage_area += area 
            else:
                status = "success"
                description = "[YOLO 模式]未发现明显裂纹"

        # ==========================================
        # 防御性编程：未知的 method
        # ==========================================
        else:
            status = "warning"
            description = f"系统预警：未知的检测方法参数 '{method}'"

        return status, description, damage_area