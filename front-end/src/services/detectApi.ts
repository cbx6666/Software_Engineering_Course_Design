import axios from "axios";
import type { DetectionResultData } from "@/types/detection";

function getBackendUrl() {
  const url = import.meta.env.VITE_BACKEND_URL as string | undefined;
  return (url ?? "").replace(/\/+$/, "");
}

const api = axios.create({
  baseURL: getBackendUrl(),
  withCredentials: true,
});

export async function detectGlassCrack(files: File[]): Promise<DetectionResultData> {
  const formData = new FormData();
  files.forEach((file) => formData.append("images", file));
  const { data } = await api.post<DetectionResultData>("/api/detect/glass-crack", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

const flatnessFieldNames = ["left_env", "left_mix", "right_env", "right_mix"] as const;
export type FlatnessFieldName = (typeof flatnessFieldNames)[number];

export async function detectGlassFlatness(filesByField: Partial<Record<FlatnessFieldName, File>>): Promise<DetectionResultData> {
  const formData = new FormData();
  flatnessFieldNames.forEach((field) => {
    const file = filesByField[field];
    if (file) formData.append(field, file);
  });

  const { data } = await api.post<DetectionResultData>("/api/detect/glass-flatness", formData);
  return data;
}


