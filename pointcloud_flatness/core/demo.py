import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from .geometry import undistort_points_cv2
from .flatness import fit_plane_least_squares, point_plane_signed_distance

def depth_from_pixels(u_left, u_right, baseline, f):
    """用像素差计算深度"""
    disparity = u_left[:,0] - u_right[:,0] + 1e-6
    Z = f * baseline / disparity
    return Z

def run_flatness_demo(
    baseline=0.11,
    Z0=0.2,
    f=800,
    noise=0.0002,
    shape_type='saddle',
    scale_height=0.002,
    color_scale=50
):
    # -----------------------------
    # 原始曲面生成
    # -----------------------------
    nx, ny = 80, 60
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    xx, yy = np.meshgrid(xs, ys)

    if shape_type == 'saddle':
        a, b = 1.0, 0.8
        zz = scale_height * (a*xx**2 - b*yy**2)
    elif shape_type == 'sin':
        zz = scale_height * 0.01*np.sin(4*np.pi*xx) * np.sin(4*np.pi*yy)
    elif shape_type == 'gaussian':
        zz = scale_height * 0.01*np.exp(-((xx/0.3)**2 + (yy/0.2)**2))
    elif shape_type == 'combo':
        a, b = 1.0, 0.8
        zz = scale_height * (a*xx**2 - b*yy**2 + 0.005*np.sin(6*np.pi*xx)*np.sin(6*np.pi*yy))
    else:
        raise ValueError("shape_type must be one of ['saddle','sin','gaussian','combo']")

    pts_raw = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # -----------------------------
    # 模拟相机投影
    # -----------------------------
    K = np.array([[f,0,320],[0,f,240],[0,0,1]], float)
    dist = np.zeros(5)

    # 左相机
    Xc_left = pts_raw.copy(); Xc_left[:,2] += Z0
    uv_left = np.zeros((pts_raw.shape[0],2))
    uv_left[:,0] = Xc_left[:,0]*K[0,0]/Xc_left[:,2] + K[0,2]
    uv_left[:,1] = Xc_left[:,1]*K[1,1]/Xc_left[:,2] + K[1,2]

    # 右相机
    Xc_right = pts_raw.copy(); Xc_right[:,0] -= baseline; Xc_right[:,2] += Z0
    uv_right = np.zeros((pts_raw.shape[0],2))
    uv_right[:,0] = Xc_right[:,0]*K[0,0]/Xc_right[:,2] + K[0,2]
    uv_right[:,1] = Xc_right[:,1]*K[1,1]/Xc_right[:,2] + K[1,2]

    # 去畸变
    norm_xy_left = undistort_points_cv2(uv_left, K, dist)

    # -----------------------------
    # 三角测量恢复 Z
    # -----------------------------
    Z = depth_from_pixels(uv_left, uv_right, baseline, f)
    Z += noise*np.random.randn(*Z.shape)

    # 恢复 X/Y
    X = (uv_left[:,0] - K[0,2])/K[0,0] * Z
    Y = (uv_left[:,1] - K[1,2])/K[1,1] * Z
    pts3d = np.c_[X, Y, Z]

    # -----------------------------
    # 使用完整点云拟合平面
    # -----------------------------
    pts_valid = pts3d.copy()
    plane_coeffs, _ = fit_plane_least_squares(pts_valid)
    dists = point_plane_signed_distance(pts_valid, plane_coeffs)
    dists_mm = dists*1000
    flatness_range = np.max(dists_mm)-np.min(dists_mm)
    flatness_rms = np.sqrt(np.mean((dists_mm - np.mean(dists_mm))**2))

    # -----------------------------
    # 坐标轴范围
    # -----------------------------
    xlim = [pts_raw[:,0].min(), pts_raw[:,0].max()]
    ylim = [pts_raw[:,1].min(), pts_raw[:,1].max()]
    zlim = [pts_raw[:,2].min(), pts_raw[:,2].max()]

    # -----------------------------
    # 可视化三图
    # -----------------------------
    fig = plt.figure(figsize=(18,6))

    # 左图
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.plot_trisurf(pts_raw[:,0], pts_raw[:,1], pts_raw[:,2], cmap='viridis', linewidth=0.2)
    ax1.set_title(f'Original Surface ({shape_type})')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_zlim(zlim)



    # 中图：使用右图点云的 scatter3D
    ax2 = fig.add_subplot(1,3,2, projection='3d')

    # 使用右图三角测量恢复的点云
    X_mid = pts_valid[:,0]
    Y_mid = pts_valid[:,1]
    Z_mid = pts_valid[:,2]

    # 平整度归一化做颜色
    norm_dists_mid = (dists_mm - np.min(dists_mm)) / (np.max(dists_mm) - np.min(dists_mm) + 1e-6)

    # 绘制 scatter3D
    sc = ax2.scatter(X_mid, Y_mid, Z_mid,
                     c=norm_dists_mid,
                     cmap='coolwarm',
                     s=10)  # 点大小可调

    ax2.set_title('3D Points via Triangulation')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # 使用右图点云范围设置坐标轴
    ax2.set_xlim(np.min(X_mid), np.max(X_mid))
    ax2.set_ylim(np.min(Y_mid), np.max(Y_mid))
    ax2.set_zlim(np.min(Z_mid), np.max(Z_mid))

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.5)
    cbar.set_label('Normalized Distance to Fitted Plane')


    # 右图：俯视平整度热图
    ax3 = fig.add_subplot(1,3,3)
    nx_grid, ny_grid = 300, 225  # 高密度网格
    Xi, Yi = np.meshgrid(np.linspace(xlim[0], xlim[1], nx_grid),
                         np.linspace(ylim[0], ylim[1], ny_grid))
    # 使用 linear 插值保证连续
    Zi = griddata((pts_valid[:,0], pts_valid[:,1]), dists_mm, (Xi, Yi), method='linear')
    im = ax3.imshow(Zi, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                    cmap='coolwarm', aspect='equal')
    fig.colorbar(im, ax=ax3, shrink=0.7, label='Distance to Fitted Plane (mm)')
    ax3.set_title('Top-down Flatness Heatmap')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')

    plt.show()

    # 输出平整度信息
    print(f"曲面类型: {shape_type}")
    print(f"拟合平面: z = {plane_coeffs[0]:.6f}x + {plane_coeffs[1]:.6f}y + {plane_coeffs[2]:.6f}")
    print(f"平整度范围: {flatness_range:.3f} mm")
    print(f"平整度 RMS: {flatness_rms:.3f} mm")

    return pts_raw, pts_valid, dists_mm
