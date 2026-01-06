import type { DetectionResultData } from "@/types/detection";
import { createApiClient } from "./http";

// 创建一个带凭证的 API 客户端实例
const api = createApiClient({ withCredentials: true });

/**
 * 历史记录项的数据结构，与后端 com.wing.glassdetect.model.History 对应
 */
export interface HistoryItem extends DetectionResultData {
  id: string;
  type: "crack" | "flatness";
  date: string; // 后端是 LocalDateTime，会被序列化为 ISO 8601 字符串
}

/**
 * 从后端获取指定用户的历史记录列表
 * @param userId 用户 ID
 * @returns 历史记录项数组
 */
export async function getHistory(userId: string): Promise<HistoryItem[]> {
  const response = await api.get<HistoryItem[]>("/api/history", {
    params: { userId },
  });
  return response.data;
}

/**
 * 从后端获取单条历史记录的详细信息
 * @param id 历史记录的 ID
 * @returns 单条历史记录项，如果未找到则可能抛出 404 错误
 */
export async function getHistoryItemById(id: string): Promise<HistoryItem> {
  const response = await api.get<HistoryItem>(`/api/history/${id}`);
  return response.data;
}
