export type BoundingBox = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
};

export function calculateBoundingBox(points: number[][]): BoundingBox {
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (const [x, y, z] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  return { minX, maxX, minY, maxY, minZ, maxZ };
}

/**
 * 缩放和平移点集到正半轴
 * 流程：1) 先缩放 2) 处理负值（平移到正半轴）3) 加 padding
 */
export function scaleAndShift(
  points: number[][],
  dists: number[],
  xyScale: number,
  zScale: number,
  padding: number = 100
): {
  transformedPoints: number[][];
  transformedDists: number[];
  bbox: BoundingBox;
} {
  const scaledPoints = points.map(([x, y, z]) => [x * xyScale, y * xyScale, z * zScale] as [number, number, number]);
  const scaledDists = dists.map((d) => d * zScale);

  const scaledBbox = calculateBoundingBox(scaledPoints);

  const shiftToPositiveX = scaledBbox.minX < 0 ? -scaledBbox.minX : 0;
  const shiftToPositiveY = scaledBbox.minY < 0 ? -scaledBbox.minY : 0;
  const shiftToPositiveZ = scaledBbox.minZ < 0 ? -scaledBbox.minZ : 0;

  const shiftX = shiftToPositiveX + padding;
  const shiftY = shiftToPositiveY + padding;
  const shiftZ = shiftToPositiveZ + padding;

  const transformedPoints = scaledPoints.map(([x, y, z]) => [x + shiftX, y + shiftY, z + shiftZ] as [number, number, number]);

  const bbox = {
    minX: scaledBbox.minX + shiftX,
    maxX: scaledBbox.maxX + shiftX,
    minY: scaledBbox.minY + shiftY,
    maxY: scaledBbox.maxY + shiftY,
    minZ: scaledBbox.minZ + shiftZ,
    maxZ: scaledBbox.maxZ + shiftZ,
  };

  return { transformedPoints, transformedDists: scaledDists, bbox };
}


