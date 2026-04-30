import numpy as np

_EPS = 1e-12


def _as_point_array(points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    return pts


def _finite_mask(points):
    return np.isfinite(points).all(axis=1)


def _fit_plane_svd(points):
    """Fit a plane by minimizing orthogonal distances."""
    pts = _as_point_array(points)
    if len(pts) < 3:
        raise ValueError("at least 3 finite points are required to fit a plane")

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    if singular_values.size < 2:
        raise ValueError("point set is degenerate and cannot define a plane")
    scale = max(float(singular_values[0]), _EPS)
    if float(singular_values[1]) <= scale * 1e-12:
        raise ValueError("point set is nearly collinear and cannot define a stable plane")

    normal = vh[-1].astype(float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= _EPS:
        raise ValueError("failed to compute a valid plane normal")
    normal /= normal_norm

    # Keep signed distances consistent with the UI: positive means larger Z
    # than the fitted reference plane.
    if normal[2] < 0:
        normal = -normal

    offset = -float(np.dot(normal, centroid))
    return normal, offset, centroid, singular_values


def _plane_to_abc(normal, offset):
    nz = float(normal[2])
    if abs(nz) <= _EPS:
        raise ValueError("plane is too close to vertical for z = ax + by + c form")
    a = -float(normal[0]) / nz
    b = -float(normal[1]) / nz
    c = -float(offset) / nz
    return (a, b, c)


def _robust_threshold(residuals, mad_thresh):
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return 0.0, 0.0

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    # 1.4826 converts MAD to a Gaussian-equivalent sigma estimate.
    threshold = float(mad_thresh) * 1.4826 * max(mad, _EPS)
    threshold = max(threshold, 1e-9)
    return med, threshold


def signed_distance_to_plane_model(points, normal, offset):
    """Signed orthogonal distance to n*x + offset = 0."""
    pts = _as_point_array(points)
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= _EPS:
        raise ValueError("normal must be non-zero")
    normal = normal / norm

    dists = np.full(pts.shape[0], np.nan, dtype=float)
    valid = _finite_mask(pts)
    dists[valid] = pts[valid] @ normal + float(offset) / norm
    return dists


def fit_plane_robust(points, mad_thresh=3.5):
    """Fit a plane with SVD, reject residual outliers by MAD, then refit."""
    pts_all = _as_point_array(points)
    valid = _finite_mask(pts_all)
    pts = pts_all[valid]
    if len(pts) < 3:
        raise ValueError("at least 3 finite points are required to fit a plane")

    normal, offset, centroid, singular_values = _fit_plane_svd(pts)
    residuals = signed_distance_to_plane_model(pts, normal, offset)
    med, threshold = _robust_threshold(residuals, mad_thresh)
    inliers = np.abs(residuals - med) <= threshold

    method = "svd"
    if np.count_nonzero(inliers) >= 3 and np.count_nonzero(~inliers) > 0:
        normal, offset, centroid, singular_values = _fit_plane_svd(pts[inliers])
        method = "svd_mad_refit"

    final_residuals = signed_distance_to_plane_model(pts, normal, offset)
    final_med, final_threshold = _robust_threshold(final_residuals, mad_thresh)
    final_inliers = np.abs(final_residuals - final_med) <= final_threshold

    residuals_all = np.full(pts_all.shape[0], np.nan, dtype=float)
    residuals_all[valid] = final_residuals

    inlier_mask_all = np.zeros(pts_all.shape[0], dtype=bool)
    inlier_mask_all[valid] = final_inliers

    return {
        "plane": _plane_to_abc(normal, offset),
        "normal": normal,
        "offset": offset,
        "centroid": centroid,
        "residuals": residuals_all,
        "inlier_mask": inlier_mask_all,
        "threshold": final_threshold,
        "singular_values": singular_values,
        "method": method,
    }


def fit_plane_least_squares(points):
    """拟合平面 z = a x + b y + c（返回 (a,b,c), unit normal）"""
    pts = _as_point_array(points)
    valid = _finite_mask(pts)
    normal, offset, _, _ = _fit_plane_svd(pts[valid])
    return _plane_to_abc(normal, offset), normal


def point_plane_signed_distance(points, plane):
    """点到平面的有符号距离（m）"""
    pts = _as_point_array(points)
    a, b, c = plane
    dists = np.full(pts.shape[0], np.nan, dtype=float)
    valid = _finite_mask(pts)
    denom = np.sqrt(a*a + b*b + 1.0)
    if denom <= _EPS:
        denom = _EPS
    dists[valid] = (pts[valid, 2] - (a * pts[valid, 0] + b * pts[valid, 1] + c)) / denom
    return dists
