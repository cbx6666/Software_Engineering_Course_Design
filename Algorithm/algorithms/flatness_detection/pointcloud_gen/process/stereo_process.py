import numpy as np
from utils.outliers import mad_mask
from utils.plane_fit import fit_plane_least_squares, point_plane_signed_distance
from utils.camera import depth_from_pixels, backproject_uv_to_xyz
from utils.interp import densify_disparity
from utils.io_utils import save_ply, export_csv, visualize_pointcloud


def process_stereo_matches(
    uv_left_sparse,
    uv_right_sparse,
    K,
    baseline,
    image_shape=None,
    densify=True,
    densify_method='cubic',
    densify_smooth_sigma=0.0,
    mad_thresh=3.5,
    export_ply_path=None,
    export_csv_path=None,
    visualize=False,
    save_fig_path=None
):
    """
    主入口流程：恢复3D、平面拟合、稠密点云、平整度指标
    """

    # ------- 稀疏匹配恢复深度 -------
    disp_sparse = uv_left_sparse[:,0] - uv_right_sparse[:,0]
    f = K[0,0]
    Z_sparse = depth_from_pixels(uv_left_sparse, uv_right_sparse, baseline, f)
    pts_sparse = backproject_uv_to_xyz(uv_left_sparse, Z_sparse, K)

    # ------- 平面拟合 -------
    valid_sparse = ~np.isnan(Z_sparse)
    pts_valid = pts_sparse[valid_sparse]

    mask = mad_mask(pts_valid[:,2], thresh=mad_thresh)
    pts_for_fit = pts_valid[mask] if np.sum(mask) >= 3 else pts_valid

    plane, normal = fit_plane_least_squares(pts_for_fit)
    dists_sparse = point_plane_signed_distance(pts_sparse, plane)

    result = {
        "pts_sparse": pts_sparse,
        "dists_sparse": dists_sparse,
        "plane_coeffs": plane,
        "normal": normal,
        "pts_dense": None,
        "dists_dense": None,
        "flatness_metrics": {}
    }

    # ------- 稠密点云 -------
    if densify:
        if image_shape is None:
            raise ValueError("要稠密化，请提供 image_shape=(h,w)")

        h, w = image_shape
        disp_map = densify_disparity(uv_left_sparse, disp_sparse, image_shape, densify_method, densify_smooth_sigma)

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        uv_flat = np.vstack([u.ravel(), v.ravel()]).T
        disp_flat = disp_map.ravel()

        valid = ~np.isnan(disp_flat) & (np.abs(disp_flat) > 1e-9)
        Z_flat = np.zeros_like(disp_flat) + np.nan
        Z_flat[valid] = f * baseline / disp_flat[valid]

        pts_flat = backproject_uv_to_xyz(uv_flat, Z_flat, K)
        pts_dense = pts_flat[~np.isnan(pts_flat).any(axis=1)]
        dists_dense = point_plane_signed_distance(pts_dense, plane)

        result["pts_dense"] = pts_dense
        result["dists_dense"] = dists_dense

        print("disp_map: nan ratio =", np.mean(np.isnan(disp_map)))
        print("num valid disp pixels:", np.sum(~np.isnan(disp_map)))
        print("pts_dense shape:", pts_dense.shape)

    # ------- 平整度指标 -------
    use_d = result["dists_dense"] if result["pts_dense"] is not None else dists_sparse
    use_d_mm = use_d * 1000

    metrics = {
        "count": len(use_d_mm),
        "flatness_range_mm": float(np.nanmax(use_d_mm) - np.nanmin(use_d_mm)),
        "flatness_rms_mm": float(np.sqrt(np.nanmean((use_d_mm - np.nanmean(use_d_mm))**2))),
        "flatness_mean_mm": float(np.nanmean(use_d_mm)),
        "flatness_std_mm": float(np.nanstd(use_d_mm)),
        "max_mm": float(np.nanmax(use_d_mm)),
        "min_mm": float(np.nanmin(use_d_mm)),
        "p95_mm": float(np.nanpercentile(np.abs(use_d_mm), 95))
    }
    result["flatness_metrics"] = metrics

    # ------- 导出 -------
    if export_ply_path and result["pts_dense"] is not None:
        save_ply(result["pts_dense"], export_ply_path)
    if export_csv_path and result["pts_dense"] is not None:
        export_csv(result["pts_dense"], result["dists_dense"], export_csv_path)
    if visualize:
        # 优先复用上面计算的 disp_map / Z_flat（如果有）
        dense_depth = None
        if densify and image_shape is not None:
            try:
                # 计算 disp_map（如果之前已计算，也可以缓存避免重复）
                disp_map = densify_disparity(
                    uv_left_sparse, disp_sparse,
                    image_shape, densify_method, densify_smooth_sigma
                )
                # 转成深度图（m）
                dense_depth = np.full_like(disp_map, np.nan, dtype=float)
                valid_mask = ~np.isnan(disp_map) & (np.abs(disp_map) > 1e-9)
                if np.any(valid_mask):
                    dense_depth[valid_mask] = f * baseline / disp_map[valid_mask]
                else:
                    dense_depth = None  # 没有有效稠密点
            except Exception as e:
                print("⚠️ 稠密化生成深度图失败：", e)
                dense_depth = None

        # 调用可视化（可接受 dense_depth 为 None）
        try:
            visualize_pointcloud(
                xyz_sparse=pts_sparse,
                dense_depth=dense_depth,
                image_shape=image_shape,
                save_path=save_fig_path
            )
            if save_fig_path:
                print(f"✅ 可视化已保存至: {save_fig_path}")
        except Exception as e:
            print("⚠️ 可视化函数执行失败：", e)

    return result
