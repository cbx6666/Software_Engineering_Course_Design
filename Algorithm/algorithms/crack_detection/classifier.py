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

    def predict_risk(self, features: Dict) -> Tuple[str, str, float]:
        """
        风险预测：结合自爆特征判断结果
        返回：status（success/warning/error）、描述、损伤面积（mm²，此处为0）
        """
        # 注释掉原损伤面积计算逻辑（若无需该参数）
        # crack_ratio = features.get("crack_ratio", 0)
        # damage_area = crack_ratio * 1000 * 1000  # 像素占比→实际面积

        # 获取自爆特征判断结果（从特征中提取）
        is_break = features.get("is_break", False)

        # 结合自爆特征判断结果决策
        if is_break:
            # 检测到自爆特征→返回错误状态
            return "error", "检测到玻璃自爆特征", 0.0
        elif features.get("crack_ratio", 0) > Config.WARNING_THRESHOLD:
            # 裂纹占比超过预警阈值→返回警告
            return "warning", "玻璃存在裂纹，有自爆风险", 0.0
        else:
            # 无异常→正常状态
            return "success", "玻璃状态正常，无自爆风险", 0.0