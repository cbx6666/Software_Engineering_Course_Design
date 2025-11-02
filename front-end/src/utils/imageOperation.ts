import { useState } from "react";
import axios from "axios";

// 返回类型
export interface DetectionResultData {
  status: "success" | "error" | "warning";
  title: string;
  description: string;
  details: { label: string; value: string }[];
}

export function useImageUpload(backendUrl: string) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // 选择图片
  const selectImage = (file: File) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl); // 释放之前 URL
    }
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // 删除图片
  const removeImage = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setImageFile(null);
    setPreviewUrl(null);
  };

  // 上传图片到后端
  const uploadImage = async (): Promise<DetectionResultData> => {
    if (!imageFile) throw new Error("没有选择图片");

    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const response = await axios.post(backendUrl, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        withCredentials: true, // 如果需要携带 Cookie
      });

      return response.data; // 假设后端返回 DetectionResultData
    } catch (err: any) {
      console.error("图片上传失败:", err);
      throw new Error(err.response?.data?.message || "上传失败");
    } finally {
      setIsUploading(false);
    }
  };

  return {
    imageFile,
    previewUrl,
    isUploading,
    selectImage,
    removeImage,
    uploadImage,
  };
}
