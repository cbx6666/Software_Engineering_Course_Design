// @ts-nocheck
/// <reference types="@react-three/fiber" />

import { useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import type { ReactThreeFiber } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

/**
 * 后端传入的点云数据结构。
 * - points/dists 单位为米，组件内部会统一转换为毫米或视觉缩放。
 * - projected_* 存在时，表示已在后端做过平面投影与距离计算。
 */
interface PointCloudData {
  points: number[][]; // N×3 array [x, y, z] (meters)
  dists: number[];    // N array, distance to plane (meters)
  plane: number[];    // [a, b, c] for z = a x + b y + c
  normal: number[];   // [nx, ny, nz]
  projected_points?: number[][];
  projected_dists?: number[];
}

/** 组件输入：仅需传入一帧点云数据即可渲染。 */
interface PointCloud3DProps {
  data: PointCloudData;
}

/**
 * 渲染点云点位。传入的 points/dists 已完成缩放与平移。
 * 使用 HSL（蓝->红）编码距离，便于肉眼区分高度偏差。
 */
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
      <pointsMaterial vertexColors size={5} sizeAttenuation={false} />
    </points>
  );
}

/**
 * 在三维空间绘制三面参考网格（XY / YZ / XZ），帮助用户感知方位与尺度。
 * 默认放置在正半轴区域，size 依据点云范围自适应。
 */
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

// 绘制坐标轴与末端标签
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
    { pos: new THREE.Vector3(0, 0, len + labelOffset), text: "Z (0.25mm)", color: "#0088ff" },
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

/** 使用 drei Html 在 3D 中叠加 2D 文本标签。 */
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

// 交互提示 + 重置按钮
function ControlsPanel({ onReset }: { onReset: () => void }) {
  return (
    <div
      className="absolute top-2 left-6 z-50 flex flex-col gap-2 pointer-events-auto"
      style={{ pointerEvents: "auto" }}
    >
      <button
        onClick={onReset}
        type="button"
        className="px-4 py-2 text-xs font-medium rounded-lg hover:bg-white/10 transition-colors"
        style={{
          background: "transparent",
          border: "1px solid rgba(255, 255, 255, 0.6)",
          color: "#ffffff",
          fontFamily: "Microsoft YaHei",
          fontWeight: 400,
          fontSize: 15,
        }}
      >
        重置视角
      </button>
    </div>
  );
}

// 只负责渲染（光源 + 点云 + 基准网格 + 坐标轴），不做坐标变换
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

/**
 * 计算点集的包围盒
 */
function calculateBoundingBox(points: number[][]): { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number } {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
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
 * 流程：1. 先缩放 2. 处理负值（平移到正半轴）3. 加 padding
 */
function scaleAndShift(
  points: number[][],
  dists: number[],
  xyScale: number,
  zScale: number,
  padding: number = 100
): {
  transformedPoints: number[][];
  transformedDists: number[];
  bbox: { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number };
} {
  // 1. 先缩放所有点
  const scaledPoints = points.map(([x, y, z]) => [
    x * xyScale,
    y * xyScale,
    z * zScale,
  ] as [number, number, number]);

  const scaledDists = dists.map(d => d * zScale);

  // 2. 计算缩放后的包围盒
  const scaledBbox = calculateBoundingBox(scaledPoints);

  // 3. 先处理负值：平移到正半轴
  const shiftToPositiveX = scaledBbox.minX < 0 ? -scaledBbox.minX : 0;
  const shiftToPositiveY = scaledBbox.minY < 0 ? -scaledBbox.minY : 0;
  const shiftToPositiveZ = scaledBbox.minZ < 0 ? -scaledBbox.minZ : 0;

  // 4. 再加 padding
  const shiftX = shiftToPositiveX + padding;
  const shiftY = shiftToPositiveY + padding;
  const shiftZ = shiftToPositiveZ + padding;

  // 5. 应用最终平移
  const transformedPoints = scaledPoints.map(([x, y, z]) => [
    x + shiftX,
    y + shiftY,
    z + shiftZ,
  ] as [number, number, number]);

  // 6. 计算最终包围盒
  const finalBbox = {
    minX: scaledBbox.minX + shiftX,
    maxX: scaledBbox.maxX + shiftX,
    minY: scaledBbox.minY + shiftY,
    maxY: scaledBbox.maxY + shiftY,
    minZ: scaledBbox.minZ + shiftZ,
    maxZ: scaledBbox.maxZ + shiftZ,
  };

  return { transformedPoints, transformedDists: scaledDists, bbox: finalBbox };
}

/**
 * 点云 3D 视图主组件。
 * - 兼容「原始点云 + 平面参数」和「后端投影结果」两种输入。
 * - 自动把坐标转换到毫米、放大 Z 差异并移到正半轴，便于观察。
 * - 提供 OrbitControls 旋转/缩放与一键重置视角。
 */
export function PointCloud3D({ data }: PointCloud3DProps) {
  // 单位：X/Y 轴使用 1mm 单位，Z 轴使用 0.25mm 单位
  const xyScale = 1000.0; // X/Y 轴：1 meter = 1000 mm
  const zScale = 4000.0;  // Z 轴：1 meter = 4000 单位（0.25mm为单位）
  const padding = 50;    // 统一 padding 值

  // useMemo 返回：
  // [0] 已缩放/平移后的点坐标
  // [1] 缩放后的距离值（用于颜色）
  // [2] 原始对齐后的包围盒
  // [3] 基准网格尺寸
  const [transformedPoints, transformedDists, bbox, guideSize] = useMemo(() => {
    let points: number[][];
    let dists: number[];

    // 如果有后端投影结果，直接使用；否则进行旋转对齐
    if (Array.isArray(data.projected_points) && Array.isArray(data.projected_dists)) {
      // 使用投影数据（已由后端处理）
      points = data.projected_points;
      dists = data.projected_dists;
    } else {
      // 计算拟合平面的旋转四元数，使其法线对齐 Z 轴
      const [a, b, c0] = data.plane;
      const normal = new THREE.Vector3(a, b, -1).normalize();
      const up = new THREE.Vector3(0, 0, 1);
      const quat = new THREE.Quaternion().setFromUnitVectors(normal, up);

      // 计算平面在旋转后的 Z 高度偏移
      const p0 = new THREE.Vector3(0, 0, c0).applyQuaternion(quat);
      const offsetZ = p0.z;

      // 旋转所有点并对齐到水平
      points = data.points.map(([x, y, z]) => {
        const raw = new THREE.Vector3(x, y, z);
        raw.applyQuaternion(quat);
        return [raw.x, raw.y, raw.z - offsetZ]; // 减去平面高度偏移
      });

      dists = data.dists;
    }

    // 统一进行缩放和平移
    const { transformedPoints, transformedDists, bbox } = scaleAndShift(
      points,
      dists,
      xyScale,
      zScale,
      padding
    );

    // 计算基准网格大小
    const sizeX = bbox.maxX - bbox.minX;
    const sizeY = bbox.maxY - bbox.minY;
    const guideSize = Math.max(sizeX, sizeY) * 1.6;

    return [transformedPoints, transformedDists, bbox, guideSize] as const;
  }, [data.points, data.dists, data.plane, data.projected_points, data.projected_dists, xyScale, zScale, padding]);

  // 相机设置（初始视角朝向正半轴）
  // bbox 已经在 scaleAndShift 中处理了平移，直接使用即可
  const cx = (bbox.maxX + bbox.minX) / 2;
  const cy = (bbox.maxY + bbox.minY) / 2;
  const cz = (bbox.maxZ + bbox.minZ) / 2;

  // 根据包围盒尺寸选择初始相机距离，保持所有内容入镜
  const camDist = Math.max(
    bbox.maxX - bbox.minX,
    bbox.maxY - bbox.minY,
    bbox.maxZ - bbox.minZ,
    1
  ) * 1.2;
  const initialCameraPos = [camDist, camDist, camDist] as [number, number, number];
  const initialTarget = [cx, cy, cz] as [number, number, number];

  const controlsRef = useRef<any>(null);

  // 交互重置：回到初始 target 与相机位置
  const handleReset = () => {
    if (controlsRef.current) {
      controlsRef.current.target.set(...initialTarget);
      controlsRef.current.object.position.set(...initialCameraPos);
      controlsRef.current.update();
    }
  };

  return (
    <div className="w-full max-w-[800px] mx-auto">
      <div
        className="w-full border rounded-lg bg-black overflow-hidden relative"
        style={{ height: "600px" }}
      >
        <ControlsPanel onReset={handleReset} />
        <Canvas
          style={{ width: "100%", height: "100%" }} // 确保跟随容器高度
          camera={{ position: initialCameraPos, fov: 45, up: [0, 0, 1] }}
        >
          <SceneContent
            points={transformedPoints}
            dists={transformedDists}
            guideSize={guideSize}
            axesSize={guideSize}
          />
          <OrbitControls
            ref={controlsRef}
            target={initialTarget}
            enablePan={false}
          />
        </Canvas>
      </div>
      {/* 颜色说明 */}
      <div className="mt-4 px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-lg">
        <div className="space-y-3">
          <div className="text-sm font-semibold text-slate-200 mb-2">颜色映射说明</div>
          <div className="flex flex-wrap items-center gap-6 text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: "#0080ff" }}></div>
              <span>蓝色：负偏差（低于标准平面）</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: "#00ff00" }}></div>
              <span>绿色：标准（接近理想平面）</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: "#ff0000" }}></div>
              <span>红色：正偏差（高于标准平面）</span>
            </div>
          </div>
          <div className="text-xs text-slate-400 mt-2 leading-relaxed">
            <p>颜色变化逻辑：点云颜色采用 HSL 色彩空间从蓝色到红色的连续渐变映射。根据各点到拟合平面的距离（偏差值），负偏差显示为蓝色，正偏差显示为红色，中间值按线性插值显示为蓝-绿-红的渐变过渡。</p>
          </div>
        </div>
      </div>
    </div>
  );
}