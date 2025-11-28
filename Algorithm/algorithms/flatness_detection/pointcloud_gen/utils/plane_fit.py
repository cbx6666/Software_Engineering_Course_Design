import numpy as np

def fit_plane_least_squares(points):
    """拟合平面 z = a x + b y + c（返回 (a,b,c), unit normal）"""
    X, Y, Z = points[:,0], points[:,1], points[:,2]
    A = np.c_[X, Y, np.ones_like(X)]
    sol, *_ = np.linalg.lstsq(A, Z, rcond=None)
    a,b,c = sol
    n = np.array([a,b,-1.0])
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm
    return (a,b,c), n

def point_plane_signed_distance(points, plane):
    """点到平面的有符号距离（m）"""
    a,b,c = plane
    num = a*points[:,0] + b*points[:,1] - points[:,2] + c
    denom = np.sqrt(a*a + b*b + 1.0)
    return num / (denom if denom != 0 else 1e-12)
