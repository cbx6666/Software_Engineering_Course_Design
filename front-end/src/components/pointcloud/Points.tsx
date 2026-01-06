// @ts-nocheck
import { useMemo } from "react";
import * as THREE from "three";

/**
 * 渲染点云点位。传入的 points/dists 已完成缩放与平移。
 * 使用 HSL（蓝->红）编码距离，便于肉眼区分高度偏差。
 */
export function Points({ points, dists }: { points: number[][]; dists: number[] }) {
  const positions = useMemo(() => new Float32Array(points.flat()), [points]);
  const colors = useMemo(() => {
    const arr = new Float32Array(points.length * 3);
    const minD = Math.min(...dists);
    const maxD = Math.max(...dists);
    const range = maxD - minD || 1;

    for (let i = 0; i < points.length; i++) {
      const t = (dists[i] - minD) / range; // 0~1
      const color = new THREE.Color().setHSL(0.66 * (1 - t), 1, 0.5); // 蓝->红
      arr[i * 3] = color.r;
      arr[i * 3 + 1] = color.g;
      arr[i * 3 + 2] = color.b;
    }
    return arr;
  }, [points.length, dists]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={5} sizeAttenuation={false} />
    </points>
  );
}


