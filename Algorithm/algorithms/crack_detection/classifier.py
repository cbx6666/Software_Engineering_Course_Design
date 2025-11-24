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