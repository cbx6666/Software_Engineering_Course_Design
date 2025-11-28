from process.stereo_process import process_stereo_matches
from utils.io_utils import load_uv_json
import numpy as np
import os
import json

def main():
    print("运行 stereo 平整度测量 ...")

    # === 实际输入 ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "result")
    os.makedirs(out_dir, exist_ok=True)
    save_fig_path = os.path.join(out_dir, "pointcloud.png")

    uv_left_path = os.path.join(data_dir, "corners_left.json")
    uv_right_path = os.path.join(data_dir, "corners_right.json")
    uv_left_sparse = load_uv_json(uv_left_path)
    uv_right_sparse = load_uv_json(uv_right_path)

    # TODO: 根据真实相机标定填写 image_shape / K / baseline
    image_shape = (480, 640)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
    baseline = 0.11

    result = process_stereo_matches(
        uv_left_sparse,
        uv_right_sparse,
        K,
        baseline,
        image_shape=image_shape,
        densify=True,
        densify_method="cubic",
        densify_smooth_sigma=1.0,
        mad_thresh=3.5,
        export_ply_path=None,
        export_csv_path=None,
        visualize=True,
        save_fig_path=save_fig_path,
    )
    metrics_path = os.path.join(out_dir, "flatness_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result["flatness_metrics"], f, indent=2, ensure_ascii=False)

    print("flatness metrics:", result["flatness_metrics"])
    print("请将真实标定参数与角点数据放入 data 目录后运行。")

if __name__ == "__main__":
    main()
