import { Points } from "@/components/pointcloud/Points";
import { CornerGuide } from "@/components/pointcloud/CornerGuide";
import { AxisArrows } from "@/components/pointcloud/AxisArrows";

export function SceneContent({
  points,
  dists,
  guideSize,
}: {
  points: number[][];
  dists: number[];
  guideSize: number;
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


