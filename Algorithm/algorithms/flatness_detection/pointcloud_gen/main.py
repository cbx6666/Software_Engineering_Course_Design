from process.stereo_process import process_stereo_matches
from utils.io_utils import load_uv_json
import numpy as np
import os
 
def main():
    # TODO: 替换为你的真实 uv_left_sparse / uv_right_sparse / K / baseline
    # 或导入 demo 数据
    print("🚀 运行 stereo 平整度测量 ...")

    # 示例（数据）
    # uv_left_sparse = load_uv_json("data/uv_left.json")
    # uv_right_sparse = load_uv_json("data/uv_right.json")
    # K = ...
    # baseline = ...
    # image_shape = (480, 640)

    # result = process_stereo_matches(...)
    # print(result["flatness_metrics"])

    # 确保使用 main.py 所在目录作为基准
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    save_fig_path = os.path.join(out_dir, "pointcloud.png")

    # === demo 数据示例 ===
    uv_left_sparse = load_uv_json("corners_left.json")
    uv_right_sparse = load_uv_json("corners_right.json")

    image_shape = (480, 640)
    K = np.array([[800,0,320],[0,800,240],[0,0,1]], float)
    baseline = 0.11
    # ======================

    result = process_stereo_matches(
        uv_left_sparse, 
        uv_right_sparse, 
        K, 
        baseline,
        image_shape=image_shape,
        densify=True,
        densify_method='cubic',
        densify_smooth_sigma=1.0,
        mad_thresh=3.5,
        export_ply_path=None,
        export_csv_path=None,
        visualize=True,
        save_fig_path=save_fig_path
    )
    print("flatness metrics:", result['flatness_metrics'])

    print("请在 main() 中填入你自己的数据后运行。")

if __name__ == "__main__":
    main()
