// @ts-nocheck
import { Html } from "@react-three/drei";
import * as THREE from "three";

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

// 绘制坐标轴与末端标签
export function AxisArrows({ size }: { size: number }) {
  const inset = 0;
  const len = size * 1.2;
  const head = size * 0.04;
  const width = head * 0.5;
  const origin = new THREE.Vector3(inset, inset, inset);
  const labelOffset = len * 0.08;
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


