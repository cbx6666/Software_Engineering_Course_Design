import { Card } from "@/components/ui/card";
import type { DetectionResultData } from "@/types/detection";
import { getStatusConfig } from "@/components/detection/status";
import { StatusHeader } from "@/components/detection/StatusHeader";
import { ResultPointCloud } from "@/components/detection/ResultPointCloud";
import { ResultImage } from "@/components/detection/ResultImage";
import { DetailsList } from "@/components/detection/DetailsList";

interface DetectionResultProps {
  result: DetectionResultData;
}

export function DetectionResult({ result }: DetectionResultProps) {
  const config = getStatusConfig(result.status);

  return (
    <Card className={`p-6 border ${config.borderColor} ${config.bgColor}`}>
      <StatusHeader result={result} />

      {/* 3D 点云可视化 */}
      {result.pointcloud && <ResultPointCloud data={result.pointcloud} />}

      {/* 平整度可视化图片 */}
      {result.image && <ResultImage src={result.image} />}

      {/* 详细指标 */}
      {result.details && <DetailsList details={result.details} />}
    </Card>
  );
}


