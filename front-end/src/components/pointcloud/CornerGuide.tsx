import * as THREE from "three";

/**
 * 在三维空间绘制三面参考网格（XY / YZ / XZ），帮助用户感知方位与尺度。
 * 默认放置在正半轴区域，size 依据点云范围自适应。
 */
export function CornerGuide({ size }: { size: number }) {
  const divs = 10;
  const half = size / 2;
  const planeColor = "#ffffff";

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


