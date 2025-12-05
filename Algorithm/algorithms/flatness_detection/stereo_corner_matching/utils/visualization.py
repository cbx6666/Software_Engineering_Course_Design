"""
可视化相关函数
"""
import cv2
import numpy as np


def visualize_matches(img_left: np.ndarray,
                      img_right: np.ndarray,
                      corners_left: np.ndarray,
                      corners_right: np.ndarray,
                      disparities: np.ndarray,
                      out_path: str) -> None:
    """生成匹配角点与视差的可视化图。"""
    if corners_left.size == 0 or corners_right.size == 0 or disparities.size == 0:
        print("  [提示] 没有可视化的匹配角点，跳过可视化。")
        return

    if img_left.ndim == 2:
        left_vis = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = img_left.copy()

    if img_right.ndim == 2:
        right_vis = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = img_right.copy()

    h = max(left_vis.shape[0], right_vis.shape[0])
    w = left_vis.shape[1] + right_vis.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:left_vis.shape[0], :left_vis.shape[1]] = left_vis
    canvas[:right_vis.shape[0], left_vis.shape[1]:left_vis.shape[1] + right_vis.shape[1]] = right_vis

    offset_x = left_vis.shape[1]
    disp_min = float(np.min(disparities))
    disp_max = float(np.max(disparities))
    disp_span = max(disp_max - disp_min, 1e-6)

    corners_left_flat = corners_left.reshape(-1, 2)
    corners_right_flat = corners_right.reshape(-1, 2)

    for idx, (pt_l, pt_r, disp) in enumerate(zip(corners_left_flat, corners_right_flat, disparities)):
        alpha = (float(disp) - disp_min) / disp_span
        color = (
            int(255 * (1.0 - alpha)),
            int(128),
            int(255 * alpha)
        )

        pt_l_int = tuple(np.round(pt_l).astype(int))
        pt_r_int = tuple(np.round(pt_r).astype(int))
        pt_r_shifted = (pt_r_int[0] + offset_x, pt_r_int[1])

        cv2.circle(canvas, pt_l_int, 5, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_r_shifted, 5, color, 2, lineType=cv2.LINE_AA)

        # 为所有点添加标号
        pair_id = str(idx + 1)
        cv2.putText(canvas, pair_id, (pt_l_int[0] + 6, pt_l_int[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, pair_id, (pt_r_shifted[0] + 6, pt_r_shifted[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)

