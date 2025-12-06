"""
角点匹配相关函数
"""
from typing import Tuple
import numpy as np


def sort_corners_to_grid(corners: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将角点按行列分配到固定大小的网格中，处理缺失角点的情况。
    
    使用统一的排序方法：
    1. 按 y 坐标排序，将角点分组为行
    2. 对每一行内的角点，按 x 坐标排序，分配到列
    3. 将角点分配到固定大小的网格中，缺失位置用 NaN 填充
    
    参数：
        corners: 角点坐标数组，形状是 (N, 1, 2)
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
    
    返回：
        (grid_corners, valid_mask)
        - grid_corners: 固定大小的角点数组，形状是 (num_rows * num_cols, 1, 2)，
                       缺失的角点位置用 NaN 填充
        - valid_mask: 布尔数组，标记哪些位置有有效的角点
    """
    pts = corners.reshape(-1, 2)  # (N, 2)
    num_rows, num_cols = pattern_size
    total_expected = num_rows * num_cols
    
    if len(pts) == 0:
        # 如果没有角点，返回全 NaN 的数组
        grid_corners = np.full((total_expected, 1, 2), np.nan, dtype=np.float32)
        valid_mask = np.zeros(total_expected, dtype=bool)
        return grid_corners, valid_mask
    
    all_x = pts[:, 0]
    all_y = pts[:, 1]
    
    # 创建固定大小的网格，初始为 NaN
    grid_corners = np.full((total_expected, 1, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros(total_expected, dtype=bool)
    
    # 步骤1：按 y 坐标排序，将角点分组为行
    y_sorted_indices = np.argsort(all_y)
    row_groups = []
    for i in range(num_rows):
        start_idx = int(i * len(y_sorted_indices) / num_rows)
        end_idx = int((i + 1) * len(y_sorted_indices) / num_rows)
        row_groups.append(y_sorted_indices[start_idx:end_idx])
    
    # 步骤2：对每一行，按 x 坐标排序，分配到列
    for row_idx, row_indices in enumerate(row_groups):
        if len(row_indices) == 0:
            continue
        
        row_x = all_x[row_indices]
        x_sorted_indices = [row_indices[i] for i in np.argsort(row_x)]
        
        # 将这一行的角点分配到列
        for col_idx in range(num_cols):
            start_idx = int(col_idx * len(x_sorted_indices) / num_cols)
            end_idx = int((col_idx + 1) * len(x_sorted_indices) / num_cols)
            if start_idx < len(x_sorted_indices):
                pt_idx = x_sorted_indices[start_idx]
                grid_idx = row_idx * num_cols + col_idx
                grid_corners[grid_idx] = corners[pt_idx]
                valid_mask[grid_idx] = True
    
    return grid_corners, valid_mask


def match_by_relative_coordinates(
    corners_left: np.ndarray, 
    corners_right: np.ndarray,
    pattern_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    角点匹配：使用网格排序方法，按行列位置匹配，处理缺失角点的情况。
    
    将左右图的角点都分配到固定大小的网格中，然后按索引匹配。
    只有两个图中都存在的角点位置才会被匹配。
    
    参数：
        corners_left: 左图角点坐标，形状是 (N, 1, 2)
        corners_right: 右图角点坐标，形状是 (M, 1, 2)
        pattern_size: 棋盘格内角点数量，格式是 (行数, 列数)
    
    返回：
        (matched_left, matched_right)
        - matched_left: 匹配成功的左图角点数组（只包含两个图都有的角点）
        - matched_right: 匹配成功的右图角点数组（只包含两个图都有的角点）
    """
    # 将角点分配到固定大小的网格中
    grid_left, valid_left = sort_corners_to_grid(corners_left, pattern_size)
    grid_right, valid_right = sort_corners_to_grid(corners_right, pattern_size)
    
    # 找到两个图都有效的角点位置
    both_valid = valid_left & valid_right
    
    # 提取匹配的角点
    matched_left = grid_left[both_valid]
    matched_right = grid_right[both_valid]
    
    num_matched = np.sum(both_valid)
    num_total = len(both_valid)
    num_missing = num_total - num_matched
    
    print(f"  角点匹配成功: {num_matched} 对（共 {num_total} 个位置，缺失 {num_missing} 个）")
    
    return matched_left, matched_right


def compute_disparities(corners_left: np.ndarray, corners_right: np.ndarray) -> np.ndarray:
    """计算视差值。"""
    x_left = corners_left[:, 0, 0]
    x_right = corners_right[:, 0, 0]
    disparities = x_left - x_right
    return disparities

