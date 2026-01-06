// @ts-nocheck
/// <reference types="@react-three/fiber" />
import { useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { PointCloudData } from "@/types/pointcloud";
import { scaleAndShift } from "@/components/pointcloud/transform";
import { SceneContent } from "@/components/pointcloud/SceneContent";
import { ControlsPanel } from "@/components/pointcloud/ControlsPanel";

interface PointCloud3DProps {
  data: PointCloudData;
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