import { PointCloud3D } from "@/components/PointCloud3D";
import type { PointCloudData } from "@/types/pointcloud";

export function ResultPointCloud({ data }: { data: PointCloudData }) {
  return (
    <div className="result-media">
      <h4>3D 点云交互视图</h4>
      <PointCloud3D data={data} />
    </div>
  );
}
