import numpy as np

def mad_mask(arr, thresh=3.5):
    """中位绝对偏差(MAD)剔除异常点"""
    valid = ~np.isnan(arr)
    if np.sum(valid) == 0:
        return np.zeros_like(arr, dtype=bool)
    med = np.median(arr[valid])
    mad = np.median(np.abs(arr[valid] - med))
    if mad < 1e-12:
        mad = 1e-12
    dev = np.abs(arr - med) / mad
    return (dev <= thresh) & valid

def fit_plane_least_squares(points):
    """拟合平面 z = ax + by + c"""
    X, Y, Z = points[:,0], points[:,1], points[:,2]
    A = np.c_[X, Y, np.ones_like(X)]
    a,b,c = np.linalg.lstsq(A, Z, rcond=None)[0]
    n = np.array([a,b,-1.0])
    n /= np.linalg.norm(n)
    return (a,b,c), n

def point_plane_signed_distance(points, plane):
    """点到平面距离"""
    a,b,c = plane
    num = a*points[:,0] + b*points[:,1] - points[:,2] + c
    denom = np.sqrt(a*a + b*b + 1)
    return num / denom
