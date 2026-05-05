export interface PointCloudData {
  /** 原始三维点坐标，单位为米 */
  points: number[][];
  /** 原始点到拟合平面的距离，单位为米 */
  dists: number[];
  /** 原始平面方程参数，格式为 z = a x + b y + c */
  plane: number[];
  /** 平面法向量 */
  normal?: number[];
  /** 后端已经投影到拟合平面坐标系下的点，前端优先使用 */
  projected_points?: number[][];
  /** 投影坐标系下的高度偏差，通常与 projected_points 的第三列一致 */
  projected_dists?: number[];
}
