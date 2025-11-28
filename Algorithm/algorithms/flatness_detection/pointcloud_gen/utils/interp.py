import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def densify_disparity(sparse_uv, sparse_disp, image_shape, method='cubic', smooth_sigma=0.0):
    """
    将稀疏视差插值到完整视差图
    """
    h, w = image_shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    points = sparse_uv[:, :2]
    values = sparse_disp
    target_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    disp_lin = griddata(points, values, target_points, method=method)
    disp_map = disp_lin.reshape((h,w))

    # 用最近邻补全 NaN
    mask_nan = np.isnan(disp_map)
    if np.any(mask_nan):
        disp_nn = griddata(points, values, target_points, method='nearest').reshape((h,w))
        disp_map[mask_nan] = disp_nn[mask_nan]

    # 可选平滑
    if smooth_sigma > 0:
        nan_mask = np.isnan(disp_map)
        filled = np.copy(disp_map)
        if np.any(~nan_mask):
            filled[nan_mask] = np.nanmedian(disp_map)
        smoothed = gaussian_filter(filled, sigma=smooth_sigma)
        smoothed[nan_mask] = np.nan
        disp_map = smoothed

    return disp_map
