// @ts-nocheck
/// <reference types="@react-three/fiber" />

import { useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import type { ReactThreeFiber } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

interface PointCloudData {
  points: number[][]; // N×3 array [x, y, z] (meters)
  dists: number[];    // N array, distance to plane (meters)
  plane: number[];    // [a, b, c] for z = a x + b y + c
  normal: number[];   // [nx, ny, nz]
  projected_points?: number[][];
  projected_dists?: number[];
}

interface PointCloud3DProps {
  data: PointCloudData;
}

function Points({ points, dists }: { points: number[][]; dists: number[] }) {
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
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial vertexColors size={2} sizeAttenuation={false} />
    </points>
  );
}

function CornerGuide({ size }: { size: number }) {
  const divs = 10;
  const half = size / 2;
  const planeColor = "#ffffff";
  const opacity = 1.0;

  return (
    <group>
      {/* XY grid at z=0 */}
      <mesh position={[half, half, 0]}>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={planeColor} side={THREE.DoubleSide} />
      </mesh>
      <gridHelper args={[size, divs, 0x999999, 0xcccccc]} position={[half, half, 0.1]} rotation={[Math.PI / 2, 0, 0]} />

      {/* YZ grid at x=0 */}
      <mesh position={[0, half, half]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={planeColor} side={THREE.DoubleSide} />
      </mesh>
      <gridHelper args={[size, divs, 0x999999, 0xcccccc]} position={[0.1, half, half]} rotation={[0, 0, Math.PI / 2]} />

      {/* XZ grid at y=0 */}
      <mesh position={[half, 0, half]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={planeColor} side={THREE.DoubleSide} />
      </mesh>
      <gridHelper args={[size, divs, 0x999999, 0xcccccc]} position={[half, 0.1, half]} rotation={[0, 0, 0]} />
    </group>
  );
}

function AxisArrows({ size }: { size: number }) {
  // 坐标轴从 (0,0,0) 开始，长度与平面一致
  const inset = 0;
  const len = size * 1.2;
  const head = size * 0.04;
  const width = head * 0.5;
  const origin = new THREE.Vector3(inset, inset, inset);
  const labelOffset = len * 0.08; // 标签与箭头末端的间距
  const labels = [
    { pos: new THREE.Vector3(len + labelOffset, 0, 0), text: "X (mm)", color: "#ff0000" },
    { pos: new THREE.Vector3(0, len + labelOffset, 0), text: "Y (mm)", color: "#00aa00" },
    { pos: new THREE.Vector3(0, 0, len + labelOffset), text: "Z (0.1mm)", color: "#0088ff" },
  ];
  return (
    <group>
      <arrowHelper args={[new THREE.Vector3(1, 0, 0), origin, len, 0xff0000, head, width]} />
      <arrowHelper args={[new THREE.Vector3(0, 1, 0), origin, len, 0x00ff00, head, width]} />
      <arrowHelper args={[new THREE.Vector3(0, 0, 1), origin, len, 0x0088ff, head, width]} />
      {labels.map((l, idx) => (
        <HtmlLabel key={idx} position={l.pos} color={l.color} text={l.text} />
      ))}
    </group>
  );
}

function HtmlLabel({ position, text, color }: { position: THREE.Vector3; text: string; color: string }) {
  return (
    <Html
      position={position.toArray()}
      style={{
        color,
        fontSize: "12px",
        fontWeight: 600,
        textShadow: "0 0 4px rgba(0,0,0,0.7)",
        whiteSpace: "nowrap",
      }}
      center
    >
      {text}
    </Html>
  );
}

function ControlsPanel({ onReset }: { onReset: () => void }) {
  return (
    <div
      className="absolute top-2 left-2 z-50 flex flex-col gap-2 pointer-events-auto"
      style={{ pointerEvents: "auto" }}
    >
      <div
        className="text-xs px-3 py-2 rounded-md shadow-lg"
        style={{
          background: "#f5f5f5", // 近乎白色，避免深色
          border: "1px solid #d4d4d8",
          color: "#0f172a",
        }}
      >
        <div className="font-semibold">操作提示</div>
        <div className="mt-1" style={{ color: "#1f2937" }}>左键：旋转</div>
        <div style={{ color: "#1f2937" }}>滚轮：缩放</div>
      </div>
      <button
        onClick={onReset}
        type="button"
        className="text-xs px-3 py-2 rounded-md shadow"
        style={{
          background: "#f8fafc",
          border: "1px solid #cbd5e1",
          color: "#0f172a",
        }}
      >
        重置视角
      </button>
    </div>
  );
}

function SceneContent({
  points,
  dists,
  guideSize,
  axesSize,
}: {
  points: number[][];
  dists: number[];
  guideSize: number;
  axesSize: number;
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[guideSize, guideSize, guideSize]} intensity={1.0} />

      <Points points={points} dists={dists} />
      <CornerGuide size={guideSize} />
      <AxisArrows size={guideSize} />
    </>
  );
}

export function PointCloud3D({ data }: PointCloud3DProps) {
  // 单位：毫米 (mm)
  // X/Y 轴：1 unit = 1 mm
  // Z 轴：根据实际高度差自适应放大，确保视觉上更明显
  const xyScale = 1000.0;

  const [transformedPoints, transformedDists, bbox, guideSize] = useMemo(() => {
    // 如果有后端投影结果，直接使用（与 Python 可视化一致）
    const useProjected = Array.isArray(data.projected_points) && Array.isArray(data.projected_dists);

    // 1) 如果有投影结果：使用投影坐标作为 points，projected_dists 作为颜色
    // 2) 否则：按原逻辑旋转、平移
    if (useProjected) {
      const projPts = data.projected_points as number[][];
      const projDists = data.projected_dists as number[];

      // 先统计原始范围
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const [x, y, z] of projPts) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
      }

      // 自适应 Z 放大
      const zRangeRaw = Math.max(maxZ - minZ, 1e-9);
      const desiredVisualRange = 4000;
      const zScale = Math.min(40000, Math.max(5000, desiredVisualRange / zRangeRaw));

      // 缩放 & 平移到正半轴
      const padding = 20;
      const shiftX = (minX < 0 ? -minX : 0) + padding;
      const shiftY = (minY < 0 ? -minY : 0) + padding;
      const shiftZ = (minZ < 0 ? -minZ : 0) + 50;

      let newMinX = Infinity, newMinY = Infinity, newMinZ = Infinity;
      let newMaxX = -Infinity, newMaxY = -Infinity, newMaxZ = -Infinity;

      const scaledPts = projPts.map(([x, y, z]) => {
        const xVal = x * xyScale + shiftX;
        const yVal = y * xyScale + shiftY;
        const zVal = z * zScale + shiftZ;
        newMinX = Math.min(newMinX, xVal);
        newMaxX = Math.max(newMaxX, xVal);
        newMinY = Math.min(newMinY, yVal);
        newMaxY = Math.max(newMaxY, yVal);
        newMinZ = Math.min(newMinZ, zVal);
        newMaxZ = Math.max(newMaxZ, zVal);
        return [xVal, yVal, zVal] as [number, number, number];
      });

      const scaledDists = projDists.map(d => d * zScale);

      const sizeX = newMaxX - newMinX;
      const sizeY = newMaxY - newMinY;
      const sizeZ = newMaxZ - newMinZ;
      const gSize = Math.max(sizeX, sizeY) * 1.2;

      return [
        scaledPts,
        scaledDists,
        { minX: newMinX, maxX: newMaxX, minY: newMinY, maxY: newMaxY, minZ: newMinZ, maxZ: newMaxZ },
        gSize
      ] as const;
    }

    // -------- 以下为原有分支（无后端投影时的旋转方案） --------
    // 1. 计算拟合平面的旋转四元数，使其法线对齐 Z 轴
    const [a, b, c0] = data.plane;
    const normal = new THREE.Vector3(a, b, -1).normalize();
    const up = new THREE.Vector3(0, 0, 1);
    const quat = new THREE.Quaternion().setFromUnitVectors(normal, up);

    // 2. 计算平面在旋转后的 Z 高度偏移
    const p0 = new THREE.Vector3(0, 0, c0).applyQuaternion(quat);
    const offsetZ = p0.z;

    const pts: number[][] = [];
    const distsScaled: number[] = [];

    // 预先记录未缩放前的 z 偏差范围，用于自适应放大
    let minZRaw = Infinity;
    let maxZRaw = -Infinity;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < data.points.length; i++) {
      // 原始点 (米)
      const raw = new THREE.Vector3(data.points[i][0], data.points[i][1], data.points[i][2]);

      // 旋转对齐到水平
      raw.applyQuaternion(quat);

      // Z 轴处理：减去平面高度(变0)，后续再按自适应比例放大
      const zRaw = raw.z - offsetZ;
      minZRaw = Math.min(minZRaw, zRaw);
      maxZRaw = Math.max(maxZRaw, zRaw);

      // X/Y 轴处理：应用毫米比例
      // 这里需要注意：旋转后的 x,y 也是米，乘 1000 变毫米
      const xVal = raw.x * xyScale;
      const yVal = raw.y * xyScale;

      // 先存储未缩放的 zRaw，稍后统一缩放
      pts.push([xVal, yVal, zRaw]);

      minX = Math.min(minX, xVal);
      maxX = Math.max(maxX, xVal);
      minY = Math.min(minY, yVal);
      maxY = Math.max(maxY, yVal);
    }

    // 根据 z 偏差范围自适应放大
    const zRangeRaw = Math.max(maxZRaw - minZRaw, 1e-9);
    const desiredVisualRange = 4000; // 期望的可视化高度范围（单位：显示单位）
    const zScale = Math.min(40000, Math.max(5000, desiredVisualRange / zRangeRaw)); // 限制范围避免过大/过小

    // 应用 zScale，并重新统计 z 范围
    minZ = Infinity; maxZ = -Infinity;
    const scaledPts = pts.map(([x, y, zRaw], idx) => {
      const zVal = zRaw * zScale;
      minZ = Math.min(minZ, zVal);
      maxZ = Math.max(maxZ, zVal);
      // 同时缩放距离用于颜色
      distsScaled.push(data.dists[idx] * zScale);
      return [x, y, zVal] as [number, number, number];
    });

    // 3. 整体平移到正半轴 (0,0,0) 起始，保留一定 padding
    const padding = 20; // padding in xy units (mm)
    const shiftX = (minX < 0 ? -minX : 0) + padding;
    const shiftY = (minY < 0 ? -minY : 0) + padding;
    // Z 轴：平移到 Z>0 区域
    const shiftZ = (minZ < 0 ? -minZ : 0) + 50;

    const shiftedPts = scaledPts.map(([x, y, z]) => [x + shiftX, y + shiftY, z + shiftZ]);

    // 更新 bbox
    const newMinX = 0, newMaxX = maxX + shiftX;
    const newMinY = 0, newMaxY = maxY + shiftY;
    const newMinZ = minZ + shiftZ, newMaxZ = maxZ + shiftZ;

    // 计算场景大小
    const sizeX = newMaxX - newMinX;
    const sizeY = newMaxY - newMinY;
    const sizeZ = newMaxZ - newMinZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);

    // 墙角网格大小：稍微比物体大一点
    const gSize = Math.max(sizeX, sizeY) * 1.2;

    return [
      shiftedPts,
      distsScaled,
      { minX: newMinX, maxX: newMaxX, minY: newMinY, maxY: newMaxY, minZ: newMinZ, maxZ: newMaxZ },
      gSize
    ] as const;

  }, [data.points, data.dists, data.plane]);

  // 相机设置（初始视角朝向正半轴）
  // 先整体平移，使所有坐标为正（含一定 padding）
  const padding = 25;
  const shiftX = (bbox.minX < 0 ? -bbox.minX : 0) + padding;
  const shiftY = (bbox.minY < 0 ? -bbox.minY : 0) + padding;
  const shiftZ = (bbox.minZ < 0 ? -bbox.minZ : 0);

  const shiftedPoints = transformedPoints.map(([x, y, z]) => [x + shiftX, y + shiftY, z + shiftZ]);

  const bboxShift = {
    minX: bbox.minX + shiftX,
    maxX: bbox.maxX + shiftX,
    minY: bbox.minY + shiftY,
    maxY: bbox.maxY + shiftY,
    minZ: bbox.minZ + shiftZ,
    maxZ: bbox.maxZ + shiftZ,
  };

  // 基准平面/坐标轴范围跟随偏移后的范围
  const guideSizeShift = Math.max(bboxShift.maxX, bboxShift.maxY) * 1.2;

  const cx = (bboxShift.maxX + bboxShift.minX) / 2;
  const cy = (bboxShift.maxY + bboxShift.minY) / 2;
  const cz = (bboxShift.maxZ + bboxShift.minZ) / 2;
  const camDist = Math.max(
    bboxShift.maxX - bboxShift.minX,
    bboxShift.maxY - bboxShift.minY,
    bboxShift.maxZ - bboxShift.minZ,
    1
  ) * 1.2;
  const initialCameraPos = [camDist, camDist, camDist] as [number, number, number];
  const initialTarget = [cx, cy, cz] as [number, number, number];

  const controlsRef = useRef<any>(null);

  const handleReset = () => {
    if (controlsRef.current) {
      controlsRef.current.target.set(...initialTarget);
      controlsRef.current.object.position.set(...initialCameraPos);
      controlsRef.current.update();
    }
  };

  return (
    <div
      className="w-full border rounded-lg bg-black overflow-hidden relative max-w-[800px] mx-auto"
      style={{ height: "600px" }}
    >
      <ControlsPanel onReset={handleReset} />
      <Canvas
        style={{ width: "100%", height: "100%" }} // 确保跟随容器高度
        camera={{ position: initialCameraPos, fov: 45, up: [0, 0, 1] }}
      >
        <SceneContent
          points={shiftedPoints}
          dists={transformedDists}
          guideSize={guideSizeShift}
          axesSize={guideSizeShift}
        />
        <OrbitControls
          ref={controlsRef}
          target={initialTarget}
          enablePan={false}
        />
      </Canvas>
    </div>
  );
}
