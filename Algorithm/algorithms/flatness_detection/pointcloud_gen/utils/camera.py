import numpy as np

def depth_from_pixels(u_left, u_right, baseline, f, eps=1e-9):
    """Z = f * baseline / disparity"""
    disparity = u_left[:,0] - u_right[:,0]
    disparity = disparity.astype(float)
    disparity[np.abs(disparity) < eps] = np.nan

    Z = np.full(disparity.shape, np.nan)
    valid = ~np.isnan(disparity)
    Z[valid] = f * baseline / disparity[valid]
    return Z

def backproject_uv_to_xyz(uv, Z, K):
    """像素 + Z -> 相机坐标 XYZ"""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    X = (uv[:,0] - cx) / fx * Z
    Y = (uv[:,1] - cy) / fy * Z

    return np.vstack([X, Y, Z]).T
