import numpy as np
import cv2

def undistort_points_cv2(uv, K, dist):
    """去畸变并转归一化坐标"""
    pts = uv.reshape(-1,1,2).astype(np.float32)
    und = cv2.undistortPoints(pts, K, dist, None, K)
    und = und.reshape(-1,2)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xn = (und[:,0] - cx) / fx
    yn = (und[:,1] - cy) / fy
    return np.stack([xn, yn], axis=1)

def compute_theta_from_norm(xn, yn):
    """计算视角"""
    return np.arctan(np.sqrt(xn*xn + yn*yn))

def depth_from_formula(norm_cam, norm_proj, baseline):
    """基线几何深度计算"""
    theta_c = compute_theta_from_norm(norm_cam[:,0], norm_cam[:,1])
    theta_p = compute_theta_from_norm(norm_proj[:,0], norm_proj[:,1])
    denom = np.sin(theta_p + theta_c)
    Z = np.full_like(theta_c, np.nan)
    mask = np.abs(denom) > 1e-9
    Z[mask] = baseline * np.sin(theta_c[mask]) / denom[mask]
    return Z

def phase_to_proj_pixel(phase_map, period=64.0):
    """相位 → 投影仪像素"""
    return (phase_map / (2*np.pi)) * period
