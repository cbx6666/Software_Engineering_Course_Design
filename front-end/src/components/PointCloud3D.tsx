// @ts-nocheck
/// <reference types="@react-three/fiber" />

import { useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { PointCloudData } from "@/types/pointcloud";

interface PointCloud3DProps {
  data: PointCloudData;
}

interface ResolvedPointCloud {
  points: number[][];
  heights: number[];
  source: "projected" | "raw";
}

interface SceneStats {
  count: number;
  minMm: number;
  maxMm: number;
  rangeMm: number;
  xSpanMm: number;
  ySpanMm: number;
  zSpanMm: number;
}

interface SceneData {
  displayPoints: number[][];
  heightsMm: number[];
  stats: SceneStats;
  gridStep: number;
  frame: {
    maxX: number;
    maxY: number;
    maxZ: number;
  };
  bounds: {
    minX: number;
    minY: number;
    minZ: number;
    maxX: number;
    maxY: number;
    maxZ: number;
    centerX: number;
    centerY: number;
    centerZ: number;
  };
  pointSize: number;
  axisLength: number;
  planePadding: number;
  cameraDistance: number;
  target: [number, number, number];
}

/** 统一解析后端返回的点云数据，优先采用算法层投影结果。 */
function resolvePointCloudData(data: PointCloudData): ResolvedPointCloud {
  if (Array.isArray(data.projected_points) && data.projected_points.length > 0) {
    const points = data.projected_points.filter(
      (point) =>
        Array.isArray(point) &&
        point.length >= 3 &&
        point.every((value) => Number.isFinite(value)),
    );
    const heights =
      Array.isArray(data.projected_dists) && data.projected_dists.length === points.length
        ? data.projected_dists.map((value) => Number(value))
        : points.map((point) => Number(point[2] ?? 0));
    return { points, heights, source: "projected" };
  }

  if (Array.isArray(data.points) && data.points.length > 0) {
    const points = data.points.filter(
      (point) =>
        Array.isArray(point) &&
        point.length >= 3 &&
        point.every((value) => Number.isFinite(value)),
    );
    const heights =
      Array.isArray(data.dists) && data.dists.length === points.length
        ? data.dists.map((value) => Number(value))
        : points.map((point) => Number(point[2] ?? 0));
    return { points, heights, source: "raw" };
  }

  return { points: [], heights: [], source: "projected" };
}

/** 按高度给点云着色，低点偏蓝，高点偏红。 */
function getHeightColor(value: number, min: number, max: number): THREE.Color {
  if (!Number.isFinite(value)) {
    return new THREE.Color("#7dd3fc");
  }

  const bound = Math.max(Math.abs(min), Math.abs(max), 1e-6);
  const normalized = THREE.MathUtils.clamp(value / bound, -1, 1);
  const color = new THREE.Color();

  if (normalized < 0) {
    const t = normalized + 1;
    color.setRGB(0.12 + 0.2 * t, 0.55 + 0.3 * t, 1.0 - 0.35 * t);
  } else {
    const t = normalized;
    color.setRGB(0.35 + 0.65 * t, 0.85 - 0.45 * t, 0.35 - 0.35 * t);
  }

  return color;
}

/** 构建用于展示的正象限点云场景数据。 */
function buildSceneData(resolved: ResolvedPointCloud): SceneData | null {
  if (resolved.points.length === 0) {
    return null;
  }

  const xMmRaw = resolved.points.map((point) => point[0] * 1000);
  const yMmRaw = resolved.points.map((point) => point[1] * 1000);
  const heightsMm = resolved.heights.map((value) => value * 1000);

  const minRawX = Math.min(...xMmRaw);
  const maxRawX = Math.max(...xMmRaw);
  const minRawY = Math.min(...yMmRaw);
  const maxRawY = Math.max(...yMmRaw);
  const minRawZ = Math.min(...heightsMm);
  const maxRawZ = Math.max(...heightsMm);

  const xSpanMm = maxRawX - minRawX;
  const ySpanMm = maxRawY - minRawY;
  const zSpanMm = maxRawZ - minRawZ;
  const dominantSpan = Math.max(xSpanMm, ySpanMm, zSpanMm, 1);
  const planePadding = Math.max(dominantSpan * 0.18, 18);

  const shiftX = planePadding + Math.max(0, -minRawX);
  const shiftY = planePadding + Math.max(0, -minRawY);
  const shiftZ = planePadding + Math.max(0, -minRawZ);

  const displayPoints = resolved.points.map((point, index) => [
    point[0] * 1000 + shiftX,
    point[1] * 1000 + shiftY,
    heightsMm[index] + shiftZ,
  ]);

  const minX = Math.min(...displayPoints.map((point) => point[0]));
  const maxX = Math.max(...displayPoints.map((point) => point[0]));
  const minY = Math.min(...displayPoints.map((point) => point[1]));
  const maxY = Math.max(...displayPoints.map((point) => point[1]));
  const minZ = Math.min(...displayPoints.map((point) => point[2]));
  const maxZ = Math.max(...displayPoints.map((point) => point[2]));

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const frameMaxX = maxX + planePadding;
  const frameMaxY = maxY + planePadding;
  const frameMaxZ = Math.max(maxZ + planePadding * 1.8, dominantSpan * 0.72 + planePadding);

  return {
    displayPoints,
    heightsMm,
    stats: {
      count: displayPoints.length,
      minMm: minRawZ,
      maxMm: maxRawZ,
      rangeMm: zSpanMm,
      xSpanMm,
      ySpanMm,
      zSpanMm,
    },
    bounds: {
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
      centerX,
      centerY,
      centerZ,
    },
    frame: {
      maxX: frameMaxX,
      maxY: frameMaxY,
      maxZ: frameMaxZ,
    },
    gridStep: getGridStep(dominantSpan),
    pointSize: THREE.MathUtils.clamp(dominantSpan / 150, 4, 7),
    axisLength: Math.max(frameMaxX, frameMaxY, frameMaxZ) * 1.08,
    planePadding,
    cameraDistance: Math.max(frameMaxX, frameMaxY, frameMaxZ) * 0.95 + planePadding * 2.2,
    target: [centerX, centerY, centerZ],
  };
}

/** 根据场景尺度选择一个合适的网格间距。 */
function getGridStep(spanMm: number): number {
  const target = Math.max(spanMm / 8, 5);
  const base = 10 ** Math.floor(Math.log10(target));
  const candidates = [1, 2, 5, 10].map((multiplier) => multiplier * base);
  return candidates.reduce((best, current) =>
    Math.abs(current - target) < Math.abs(best - target) ? current : best,
  );
}

/** 渲染点云本体。 */
function PointCloudPoints({
  points,
  heightsMm,
  pointSize,
}: {
  points: number[][];
  heightsMm: number[];
  pointSize: number;
}) {
  const positions = useMemo(() => new Float32Array(points.flat()), [points]);
  const colors = useMemo(() => {
    const arr = new Float32Array(points.length * 3);
    const minHeight = Math.min(...heightsMm);
    const maxHeight = Math.max(...heightsMm);

    for (let i = 0; i < points.length; i += 1) {
      const color = getHeightColor(heightsMm[i], minHeight, maxHeight);
      arr[i * 3] = color.r;
      arr[i * 3 + 1] = color.g;
      arr[i * 3 + 2] = color.b;
    }

    return arr;
  }, [points.length, heightsMm]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={pointSize} sizeAttenuation={false} />
    </points>
  );
}

/** 在三维场景中标注坐标轴文字。 */
function AxisLabel({
  position,
  text,
  color,
}: {
  position: [number, number, number];
  text: string;
  color: string;
}) {
  return (
    <Html
      position={position}
      center
      style={{
        color,
        fontSize: "12px",
        fontWeight: 700,
        whiteSpace: "nowrap",
        textShadow: "0 0 4px rgba(0, 0, 0, 0.65)",
        pointerEvents: "none",
      }}
    >
      {text}
    </Html>
  );
}

/** 在指定平面上绘制均匀网格线，不保留中心十字主线。 */
function PlaneGrid({
  width,
  height,
  step,
  orientation,
  color = "#83f0bf",
}: {
  width: number;
  height: number;
  step: number;
  orientation: "xy" | "xz" | "yz";
  color?: string;
}) {
  const segments = useMemo(() => {
    const vertices: number[] = [];
    const pushLine = (start: [number, number, number], end: [number, number, number]) => {
      vertices.push(...start, ...end);
    };

    const xLines = Math.max(1, Math.round(width / step));
    const yLines = Math.max(1, Math.round(height / step));

    for (let index = 0; index <= xLines; index += 1) {
      const value = Math.min(index * step, width);
      if (orientation === "xy") {
        pushLine([value, 0, 0], [value, height, 0]);
      } else if (orientation === "xz") {
        pushLine([value, 0, 0], [value, 0, height]);
      } else {
        pushLine([0, value, 0], [0, value, height]);
      }
    }

    for (let index = 0; index <= yLines; index += 1) {
      const value = Math.min(index * step, height);
      if (orientation === "xy") {
        pushLine([0, value, 0], [width, value, 0]);
      } else if (orientation === "xz") {
        pushLine([0, 0, value], [width, 0, value]);
      } else {
        pushLine([0, 0, value], [0, width, value]);
      }
    }

    return new Float32Array(vertices);
  }, [height, orientation, step, width]);

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[segments, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color={color} transparent opacity={0.28} />
    </lineSegments>
  );
}

/** 绘制围住点云的三个半透明参考平面。 */
function ReferencePlanes({
  bounds,
  frame,
  axisLength,
  gridStep,
}: {
  bounds: SceneData["bounds"];
  frame: SceneData["frame"];
  axisLength: number;
  gridStep: number;
}) {
  const planeColor = "#6ee7b7";
  const xyWidth = frame.maxX;
  const xyHeight = frame.maxY;
  const yzHeight = frame.maxY;
  const yzDepth = frame.maxZ;
  const xzWidth = frame.maxX;
  const xzDepth = frame.maxZ;

  return (
    <group>
      <mesh position={[xyWidth / 2, xyHeight / 2, 0]}>
        <planeGeometry args={[xyWidth, xyHeight]} />
        <meshBasicMaterial color={planeColor} transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      <PlaneGrid width={xyWidth} height={xyHeight} step={gridStep} orientation="xy" />

      <mesh position={[xzWidth / 2, 0, xzDepth / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[xzWidth, xzDepth]} />
        <meshBasicMaterial color={planeColor} transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      <PlaneGrid width={xzWidth} height={xzDepth} step={gridStep} orientation="xz" />

      <mesh position={[0, yzHeight / 2, yzDepth / 2]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[yzDepth, yzHeight]} />
        <meshBasicMaterial color={planeColor} transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      <PlaneGrid width={yzHeight} height={yzDepth} step={gridStep} orientation="yz" />

      <arrowHelper
        args={[
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(0, 0, 0),
          axisLength,
          0xef4444,
          axisLength * 0.08,
          axisLength * 0.045,
        ]}
      />
      <arrowHelper
        args={[
          new THREE.Vector3(0, 1, 0),
          new THREE.Vector3(0, 0, 0),
          axisLength,
          0x22c55e,
          axisLength * 0.08,
          axisLength * 0.045,
        ]}
      />
      <arrowHelper
        args={[
          new THREE.Vector3(0, 0, 1),
          new THREE.Vector3(0, 0, 0),
          axisLength,
          0x3b82f6,
          axisLength * 0.08,
          axisLength * 0.045,
        ]}
      />

      <AxisLabel position={[axisLength * 1.1, 0, 0]} text="X' (mm)" color="#ef4444" />
      <AxisLabel position={[0, axisLength * 1.1, 0]} text="Y' (mm)" color="#22c55e" />
      <AxisLabel position={[0, 0, axisLength * 1.1]} text="Z' (mm)" color="#3b82f6" />
    </group>
  );
}

/** 组装三维场景中的灯光、参考平面和点云。 */
function SceneContent({
  sceneData,
}: {
  sceneData: SceneData;
}) {
  return (
    <>
      <ambientLight intensity={0.72} />
      <directionalLight
        position={[
          sceneData.frame.maxX * 1.15,
          -sceneData.frame.maxY * 0.72,
          sceneData.frame.maxZ * 1.12,
        ]}
        intensity={1.15}
      />
      <directionalLight
        position={[
          sceneData.frame.maxX * 0.28,
          sceneData.frame.maxY * 1.08,
          sceneData.frame.maxZ * 0.78,
        ]}
        intensity={0.45}
      />

      <ReferencePlanes
        bounds={sceneData.bounds}
        frame={sceneData.frame}
        axisLength={sceneData.axisLength}
        gridStep={sceneData.gridStep}
      />
      <PointCloudPoints
        points={sceneData.displayPoints}
        heightsMm={sceneData.heightsMm}
        pointSize={sceneData.pointSize}
      />
    </>
  );
}

/** 提供常用视角切换按钮。 */
function ViewToolbar({
  onReset,
  onFront,
  onTop,
  onSide,
}: {
  onReset: () => void;
  onFront: () => void;
  onTop: () => void;
  onSide: () => void;
}) {
  return (
    <div className="pointcloud-toolbar" style={{ pointerEvents: "auto" }}>
      <button type="button" className="pointcloud-reset" onClick={onReset}>
        重置视角
      </button>
      <button type="button" className="pointcloud-reset" onClick={onFront}>
        正视
      </button>
      <button type="button" className="pointcloud-reset" onClick={onTop}>
        俯视
      </button>
      <button type="button" className="pointcloud-reset" onClick={onSide}>
        侧视
      </button>
    </div>
  );
}

export function PointCloud3D({ data }: PointCloud3DProps) {
  const resolved = useMemo(() => resolvePointCloudData(data), [data]);
  const sceneData = useMemo(() => buildSceneData(resolved), [resolved]);
  const controlsRef = useRef<any>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  /** 统一设置相机位置并保持观察中心落在点云中部。 */
  const setCameraView = (position: [number, number, number]) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls || !sceneData) {
      return;
    }

    camera.position.set(...position);
    controls.target.set(...sceneData.target);
    controls.update();
  };

  if (!sceneData) {
    return (
      <div className="result-media">
        <h4>3D 点云交互视图</h4>
        <div className="result-media__frame">暂无可显示的点云数据</div>
      </div>
    );
  }

  const initialCameraPos: [number, number, number] = [
    sceneData.frame.maxX + sceneData.cameraDistance * 0.42,
    -sceneData.cameraDistance * 0.7,
    sceneData.frame.maxZ + sceneData.cameraDistance * 0.55,
  ];

  const frontCameraPos: [number, number, number] = [
    sceneData.target[0],
    -sceneData.cameraDistance,
    sceneData.target[2] + sceneData.cameraDistance * 0.18,
  ];

  const topCameraPos: [number, number, number] = [
    sceneData.target[0],
    sceneData.target[1],
    sceneData.frame.maxZ + sceneData.cameraDistance,
  ];

  const sideCameraPos: [number, number, number] = [
    sceneData.frame.maxX + sceneData.cameraDistance,
    sceneData.target[1],
    sceneData.target[2] + sceneData.cameraDistance * 0.18,
  ];

  return (
    <div className="pointcloud-shell">
      <div className="pointcloud-canvas-wrap">
        <ViewToolbar
          onReset={() => setCameraView(initialCameraPos)}
          onFront={() => setCameraView(frontCameraPos)}
          onTop={() => setCameraView(topCameraPos)}
          onSide={() => setCameraView(sideCameraPos)}
        />

        <Canvas
          style={{ width: "100%", height: "100%" }}
          camera={{ position: initialCameraPos, fov: 40, up: [0, 0, 1] }}
          onCreated={({ camera }) => {
            cameraRef.current = camera as THREE.PerspectiveCamera;
          }}
        >
          <SceneContent sceneData={sceneData} />
          <OrbitControls
            ref={controlsRef}
            target={sceneData.target}
            enablePan
            enableZoom
            enableRotate
            enableDamping
            dampingFactor={0.08}
            screenSpacePanning
          />
        </Canvas>
      </div>

      <div className="pointcloud-legend">
        <h4>点云说明</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-swatch" style={{ backgroundColor: "#3b82f6", color: "#3b82f6" }}></div>
            <span>蓝色：低于拟合平面</span>
          </div>
          <div className="legend-item">
            <div className="legend-swatch" style={{ backgroundColor: "#84cc16", color: "#84cc16" }}></div>
            <span>绿色：接近拟合平面</span>
          </div>
          <div className="legend-item">
            <div className="legend-swatch" style={{ backgroundColor: "#ef4444", color: "#ef4444" }}></div>
            <span>红色：高于拟合平面</span>
          </div>
        </div>
        <p className="legend-note">
          点数 {sceneData.stats.count}，高度范围 {sceneData.stats.minMm.toFixed(2)} mm 到{" "}
          {sceneData.stats.maxMm.toFixed(2)} mm，跨度 {sceneData.stats.rangeMm.toFixed(2)} mm；平面尺寸约{" "}
          {sceneData.stats.xSpanMm.toFixed(1)} mm × {sceneData.stats.ySpanMm.toFixed(1)} mm，Z 向范围约{" "}
          {sceneData.stats.zSpanMm.toFixed(2)} mm。
        </p>
        {resolved.source === "raw" && (
          <p className="legend-note">
            当前结果未提供算法层投影坐标，前端暂时使用原始点云字段显示；建议以后端投影结果为准。
          </p>
        )}
      </div>
    </div>
  );
}
