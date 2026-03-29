import { PointCloud3D } from "@/components/PointCloud3D";
import type { PointCloudData } from "@/types/pointcloud";

export function ResultPointCloud({ data }: { data: PointCloudData }) {
  return (
    <div className="mb-6">
      <h4 className="text-white text-sm font-medium mb-3">3D 点云交互视图</h4>
      <PointCloud3D data={data} />
    </div>
  );
}


