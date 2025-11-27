import numpy as np

def mad_mask(arr, thresh=3.5):
    """中位绝对偏差(MAD)剔除异常点；返回有效点布尔掩码"""
    valid = ~np.isnan(arr)
    if np.sum(valid) == 0:
        return np.zeros_like(arr, dtype=bool)
    med = np.median(arr[valid])
    mad = np.median(np.abs(arr[valid] - med))
    if mad < 1e-12:
        mad = 1e-12
    dev = np.abs(arr - med) / mad
    return (dev <= thresh) & valid
