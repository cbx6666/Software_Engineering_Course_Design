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
    print(f"✅ 已输出PLY文件: {filename}")

def export_csv(points, dists, filename):
    df = pd.DataFrame({
        'X': points[:,0],
        'Y': points[:,1],
        'Z': points[:,2],
        'Dist_to_plane_m': dists
    })
    df.to_csv(filename, index=False)
    print(f"✅ 已输出CSV文件: {filename}")

def visualize_pointcloud(
        xyz_sparse,
        dense_depth=None,
        save_path=None,
        image_shape=None,
        cmap='jet'
    ):
    """
    生成 2 子图比较：
    左：原始稀疏点云（3D）
    右：稠密深度热力图（2D）
    """

    fig = plt.figure(figsize=(12, 6))

    # -----------------------
    # 1. 左：稀疏点云 (3D)
    # -----------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(
        xyz_sparse[:, 0],
        xyz_sparse[:, 1],
        xyz_sparse[:, 2],
        s=8, c='blue'
    )
    ax1.set_title("Sparse 3D Points (Original)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=35, azim=35)

    # -----------------------
    # 2. 右：稠密深度热图 (2D)
    # -----------------------
    ax2 = fig.add_subplot(1, 2, 2)
    if dense_depth is not None and image_shape is not None:
        dense_depth_img = dense_depth.reshape(image_shape)
        im = ax2.imshow(dense_depth_img, cmap=cmap)
        ax2.set_title("Dense Depth (Interpolated)")
        fig.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, "No dense depth available", ha='center')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    return fig