import type { DetectionResultData } from "@/types/detection";

export type DetectionType = "crack" | "flatness";

export interface HistoryItem extends DetectionResultData {
  id: string;
  type: DetectionType;
  date: string;
  previewImage?: string; // a small thumbnail
}

// Simulate some delay
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const mockHistoryData: HistoryItem[] = [
  {
    id: "1",
    type: "crack",
    date: "2024-05-20T10:30:00Z",
    status: "success",
    title: "办公楼A座-幕墙检测",
    description: "检测到3处微小裂纹",
    previewImage: "/placeholder/crack-thumb.jpg",
    details: [
      { label: "检测区域", value: "A区-左下角" },
      { label: "裂纹数量", value: "3" },
      { label: "最大裂纹长度", value: "2.1 cm" },
    ],
  },
  {
    id: "2",
    type: "flatness",
    date: "2024-05-19T14:00:00Z",
    status: "warning",
    title: "研发中心-外墙平整度分析",
    description: "部分区域平整度超出预警阈值",
    previewImage: "/placeholder/flatness-thumb.jpg",
    details: [
      { label: "最大偏差", value: "+3.5 mm" },
      { label: "主要影响区域", value: "顶部横梁" },
    ],
  },
  {
    id: "3",
    type: "crack",
    date: "2024-05-18T09:00:00Z",
    status: "error",
    title: "仓库B-玻璃门检测",
    description: "图像质量过低，无法完成检测",
    previewImage: "/placeholder/error-thumb.jpg",
    details: [{ label: "失败原因", value: "图像模糊且存在强反光" }],
  },
  {
    id: "4",
    type: "flatness",
    date: "2024-05-17T16:20:00Z",
    status: "success",
    title: "实验楼-西侧幕墙平整度",
    description: "平整度符合标准",
    previewImage: "/placeholder/flatness-thumb-2.jpg",
    details: [{ label: "最大偏差", value: "-0.8 mm" }],
  },
];

export async function getHistory(userId: string): Promise<HistoryItem[]> {
  console.log(`Fetching history for user ${userId}...`);
  await sleep(800); // Simulate network latency
  return Promise.resolve(mockHistoryData);
}
