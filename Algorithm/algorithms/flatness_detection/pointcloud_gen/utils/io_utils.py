import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


def load_uv_json(filename):
    """
    读取稀疏匹配点 JSON 文件，返回 N x 2 numpy array
    JSON 格式：
    [
        [u0, v0],
        [u1, v1],
        ...
    ]
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # main.py 中 data 文件夹的路径 = 当前目录的上一级 + /data/
    project_root = os.path.dirname(base_dir)
    data_path = os.path.join(project_root, "data", filename)  # 注意：filename 只写文件名

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    with open(data_path, "r") as f:
        return np.array(json.load(f), dtype=float)


def save_ply(points_xyz, filename):
    verts = np.array([tuple(p) for p in points_xyz], dtype=[('x','f4'),('y','f4'),('z','f4')])
    element = PlyElement.describe(verts, 'vertex')
    PlyData([element]).write(filename)
    print(f"已输出PLY文件: {filename}")

def export_csv(points, dists, filename):
    df = pd.DataFrame({
        'X': points[:,0],
        'Y': points[:,1],
        'Z': points[:,2],
        'Dist_to_plane_m': dists
    })
    df.to_csv(filename, index=False)
    print(f"已输出CSV文件: {filename}")


def project_to_plane_normal(points):
    """
    输入：N×3 稀疏点云
    输出：
        projected_points: 在新的( x', y', z' ) 坐标
        normal: 拟合平面法线 (单位向量)
    """

    # 去均值
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # 使用 SVD 最小二乘拟合平面
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]        # 最小奇异值对应法向量
    normal = normal / np.linalg.norm(normal)

    # 构造新坐标系
    # 保证法线不与 z 轴平行
    world_z = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(normal, world_z)) > 0.9:
        world_z = np.array([1, 0, 0], dtype=float)

    x_axis = np.cross(world_z, normal)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 构造旋转矩阵
    R = np.vstack([x_axis, y_axis, normal]).T   # 3×3

    # 投影
    projected = centered @ R

    return projected, normal

def visualize_pointcloud(
        xyz_sparse,
        dense_depth=None,
        save_path=None,
        image_shape=None,
        cmap='jet'
    ):
    """
    3 图：
    1. 原始 3D 点云（颜色 = Z）
    2. 法线重投影 3D 点云（颜色 = Z'）
    3. 法线坐标系下的平面散点图 (X', Y')，颜色 = Z'
    """

    # ======== 平面拟合 + 投影 ========
    projected_pts, plane_normal = project_to_plane_normal(xyz_sparse)
    z_new = projected_pts[:, 2]

    x = xyz_sparse[:, 0]
    y = xyz_sparse[:, 1]
    z = xyz_sparse[:, 2]

    fig = plt.figure(figsize=(18, 6))


    # =====================================================
    # 1. 原始 3D 稀疏点云
    # =====================================================
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    sc1 = ax1.scatter(x, y, z, s=10, c=z, cmap=cmap)
    fig.colorbar(sc1, ax=ax1, shrink=0.5)
    ax1.set_title("Original 3D Sparse Points (Colored by Z)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=35, azim=35)


    # =====================================================
    # 2. 法线重投影 3D 点云（主图）
    # =====================================================
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    sc2 = ax2.scatter(
        projected_pts[:, 0],
        projected_pts[:, 1],
        projected_pts[:, 2],
        s=10,
        c=z_new,
        cmap=cmap
    )
    fig.colorbar(sc2, ax=ax2, shrink=0.5)
    ax2.set_title("3D Points in Plane-Normal Coordinates (Colored by Z')")
    ax2.set_xlabel("X'")
    ax2.set_ylabel("Y'")
    ax2.set_zlabel("Z'")
    ax2.view_init(elev=35, azim=35)


    # =====================================================
    # 3. 平面散点图 + 等高线（重点增强）
    # =====================================================
    ax3 = fig.add_subplot(1, 3, 3)

    # 散点
    sc3 = ax3.scatter(
        projected_pts[:, 0],
        projected_pts[:, 1],
        s=15,
        c=z_new,
        cmap=cmap
    )
    fig.colorbar(sc3, ax=ax3, shrink=0.5)

    # ===== 添加等高线 =====
    # 计算范围，添加边距以确保包含所有边缘点
    x_min, x_max = np.min(projected_pts[:,0]), np.max(projected_pts[:,0])
    y_min, y_max = np.min(projected_pts[:,1]), np.max(projected_pts[:,1])
    
    # 计算范围大小
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 添加 10% 的边距，确保边缘点被包含
    padding_x = x_range * 0.1 if x_range > 0 else 0.01
    padding_y = y_range * 0.1 if y_range > 0 else 0.01
    
    x_min_padded = x_min - padding_x
    x_max_padded = x_max + padding_x
    y_min_padded = y_min - padding_y
    y_max_padded = y_max + padding_y
    
    # 生成规则网格（使用带边距的范围）
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min_padded, x_max_padded, 200),
        np.linspace(y_min_padded, y_max_padded, 200)
    )

    # 用 griddata 基于散点插值得到 Z' 网格
    from scipy.interpolate import griddata
    grid_z = griddata(
        projected_pts[:, 0:2],
        z_new,
        (grid_x, grid_y),
        method='cubic'
    )

    # 等高线
    cs = ax3.contour(
        grid_x, grid_y, grid_z,
        levels=12,               # 自动 12 条等高线
        linewidths=0.8,
        colors='k',
        alpha=0.7
    )

    # 添加等高线文本标签
    ax3.clabel(cs, inline=True, fontsize=8, fmt="%.4f")

    ax3.set_title("Projected XY' + Contour Lines (Colored by Z')")
    ax3.set_xlabel("X'")
    ax3.set_ylabel("Y'")
    ax3.set_aspect("equal", "box")
    
    # 设置坐标轴范围，确保显示所有点（包括边距）
    ax3.set_xlim(x_min_padded, x_max_padded)
    ax3.set_ylim(y_min_padded, y_max_padded)


    plt.tight_layout()


    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    return fig
