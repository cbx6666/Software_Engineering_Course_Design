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