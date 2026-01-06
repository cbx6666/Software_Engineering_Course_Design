import type { PointCloudData } from "@/types/pointcloud";

export type DetectionStatus = "success" | "warning" | "error";

export interface DetectionDetail {
  label: string;
  value: string;
  description?: string;
  image?: string;
}

export interface DetectionResultData {
  status: DetectionStatus;
  title: string;
  description: string;
  details?: DetectionDetail[];
  /** 平整度可视化图片（base64 data URI） */
  image?: string;
  pointcloud?: PointCloudData;
}


