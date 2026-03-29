export interface PointCloudData {
  /** N×3 array [x, y, z] (meters) */
  points: number[][];
  /** N array, distance to plane (meters) */
  dists: number[];
  /** [a, b, c] for z = a x + b y + c */
  plane: number[];
  /** [nx, ny, nz] */
  normal?: number[];
  /** Optional: already projected by backend */
  projected_points?: number[][];
  projected_dists?: number[];
}


