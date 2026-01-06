import type { DetectionResultData } from "@/types/detection";
import { createApiClient, postFormData } from "./http";

const api = createApiClient({ withCredentials: true });

export async function detectGlassCrack(userId: string, files: File[]): Promise<DetectionResultData> {
  const formData = new FormData();
  formData.append("userId", userId);
  files.forEach((file) => formData.append("images", file));
  return await postFormData<DetectionResultData>(api, "/api/detect/glass-crack", formData);
}

const flatnessFieldNames = ["left_env", "left_mix", "right_env", "right_mix"] as const;
export type FlatnessFieldName = (typeof flatnessFieldNames)[number];

export async function detectGlassFlatness(
  userId: string,
  filesByField: Partial<Record<FlatnessFieldName, File>>
): Promise<DetectionResultData> {
  const formData = new FormData();
  formData.append("userId", userId);
  flatnessFieldNames.forEach((field) => {
    const file = filesByField[field];
    if (file) formData.append(field, file);
  });

  return await postFormData<DetectionResultData>(api, "/api/detect/glass-flatness", formData);
}
