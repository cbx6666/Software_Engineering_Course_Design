import json
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_ply(points_xyz, filename):
    """保存点云为PLY文件"""
    verts = np.array([tuple(p) for p in points_xyz], dtype=[('x','f4'),('y','f4'),('z','f4')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el]).write(filename)
    print(f"✅ 已输出PLY文件: {filename}")

def export_csv(points_world, dists, out_csv):
    """保存点云与平整度数据为CSV"""
    df = pd.DataFrame({
        'X': points_world[:,0],
        'Y': points_world[:,1],
        'Z': points_world[:,2],
        'Dist_to_plane_m': dists
    })
    df.to_csv(out_csv, index=False)
    print(f"✅ 已输出CSV文件: {out_csv}")
