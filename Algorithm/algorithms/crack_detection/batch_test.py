import os
import glob
import shutil
from .breakage_algorithm import GlassBreakageAlgorithm
from .config import Config

# ================= 配置区域 =================
# 测试数据集路径配置（YOLO格式）
YOLO_IMAGES_DIR = "./datasets/etest/images"
YOLO_LABELS_DIR = "./datasets/etest/labels"

# 定义YOLO标签
# 数据集 ID=0 代表裂纹
ABNORMAL_CLASS_IDS = [0] 
# 检测方法 ('traditional', 'yolo', 'combined'暂时空缺)
TEST_METHOD = 'traditional'
# ===========================================

def parse_yolo_label(label_path):
    """解析YOLO格式标签文件"""
    if not os.path.exists(label_path):
        return False
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            class_id = int(parts[0])
            if class_id in ABNORMAL_CLASS_IDS:
                return True
    return False

def move_error_sample(img_path, label_path):
    """
    将出错的样本【移动】到指定文件夹
    """
    filename = os.path.basename(img_path)
    dst_img_path = os.path.join(Config.REMOVE_IMAGE_DIR, filename)
    
    # 1. 移动图片
    # 如果目标文件夹已有同名文件，shutil.move 可能会报错或覆盖，建议先检查
    if os.path.exists(dst_img_path):
        print(f"  ⚠️ 目标文件夹已存在 {filename}，跳过移动。")
    else:
        shutil.move(img_path, dst_img_path)
        print(f"  └── 🚚 图片已移动: {filename}")
    
    # 2. 移动标签 (如果存在的话)
    if os.path.exists(label_path):
        label_filename = os.path.basename(label_path)
        dst_label_path = os.path.join(Config.REMOVE_LABEL_DIR, label_filename)
        
        if os.path.exists(dst_label_path):
            pass # 目标标签已存在，跳过
        else:
            shutil.move(label_path, dst_label_path)

def batch_test_yolo():
    """针对YOLO数据集的批量测试"""
    # 仅当开启移动功能时，才创建目标文件夹
    if Config.IS_MOVE_ERROR_SAMPLES:
        os.makedirs(Config.REMOVE_IMAGE_DIR, exist_ok=True)
        os.makedirs(Config.REMOVE_LABEL_DIR, exist_ok=True)
        print(f"📂 错误样本将被【移动】至: \n  - {Config.REMOVE_IMAGE_DIR}\n  - {Config.REMOVE_LABEL_DIR}")
    else:
        print("🔒 错误样本移动功能已关闭 (IS_MOVE_ERROR_SAMPLES=False)")

    algorithm = GlassBreakageAlgorithm()
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    image_files = glob.glob(os.path.join(YOLO_IMAGES_DIR, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    total = len(image_files)
    print(f"🚀 开始测试，共找到 {total} 张图片...")
    print("-" * 60)

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # 1. 获取 Ground Truth
        label_path = os.path.join(YOLO_LABELS_DIR, f"{basename}.txt")
        is_actually_abnormal = parse_yolo_label(label_path)
        true_label_str = "异常" if is_actually_abnormal else "正常"

        try:
            # 2. 运行算法
            result = algorithm.run(img_path, method=TEST_METHOD)
            
            # 3. 解析预测结果
            status = result.get("status", "success")
            is_predicted_abnormal = status in ["warning", "error"]
            
            pred_label_str = "异常" if is_predicted_abnormal else "正常"
            
            # 4. 统计逻辑
            is_error = False
            status_icon = ""

            if is_actually_abnormal and is_predicted_abnormal:
                TP += 1
                status_icon = "✅ TP"
            elif not is_actually_abnormal and not is_predicted_abnormal:
                TN += 1
                status_icon = "✅ TN"
            elif not is_actually_abnormal and is_predicted_abnormal:
                FP += 1
                status_icon = "❌ 误报(FP)"
                is_error = True
            elif is_actually_abnormal and not is_predicted_abnormal:
                FN += 1
                status_icon = "❌ 漏报(FN)"
                is_error = True
            
            # 终端输出
            if Config.IS_PRINT or is_error:
                print(f"[{i+1}/{total}] {filename} | 真实:{true_label_str} 预测:{pred_label_str} -> {status_icon}")
            
            # 仅当出错 且 开关打开时，才移动文件
            if is_error and Config.IS_MOVE_ERROR_SAMPLES:
                move_error_sample(img_path, label_path)

        except Exception as e:
            print(f"[{i+1}/{total}] {filename} 运行出错: {e}")

    # ================= 计算指标 =================
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"样本总数: {total}")
    print(f"TP (正确检测裂纹): {TP}")
    print(f"TN (正确识别正常): {TN}")
    print(f"FP (误报): {FP}")
    print(f"FN (漏报): {FN}")
    print("-" * 30)
    print(f"准确率: {accuracy:.2%}") # 正确判断数总占比
    print(f"精确率: {precision:.2%}") # 捕获的裂纹图像中正确判断多少占比
    print(f"召回率: {recall:.2%}") # 裂纹图像中成功捕获多少占比
    print(f"F1:    {f1_score:.2f}") # 精确率和召回率的调和平均数
    
    if Config.IS_MOVE_ERROR_SAMPLES:
        print(f"\n⚠️ 注意：错误样本已【移动】出原文件夹，下次测试总数将减少！")
        print(f"📂 移动位置: {Config.REMOVE_IMAGE_DIR}")
    else:
        print(f"\n💡 提示：移动功能未开启，原数据集保持不变。")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs(Config.CANNY_SAVE_PATH, exist_ok=True)
    batch_test_yolo()