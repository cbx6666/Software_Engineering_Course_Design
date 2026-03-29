import { useState } from "react";
import axios from "axios";

import type { DetectionResultData } from "@/types/detection";

export function useImageUpload(backendUrl: string) {
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // 选择图片
  const selectImages = (files: File[]) => {
    previewUrls.forEach(url => URL.revokeObjectURL(url));

    const urls = files.map(file => URL.createObjectURL(file));
    setImageFiles(files);
    setPreviewUrls(urls);
  };

  // 删除图片
  const removeImage = (index: number) => {
    URL.revokeObjectURL(previewUrls[index]);
    const newFiles = imageFiles.filter((_, i) => i !== index);
    const newUrls = previewUrls.filter((_, i) => i !== index);

    setImageFiles(newFiles);
    setPreviewUrls(newUrls);
  };

  // 上传图片到后端
  const uploadImage = async (): Promise<DetectionResultData> => {
    if (imageFiles.length === 0) throw new Error("没有选择图片");

    setIsUploading(true);

    const formData = new FormData();
    imageFiles.forEach(file => formData.append("images", file));

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
    imageFiles,
    previewUrls,
    isUploading,
    selectImages,
    removeImage,
    uploadImage,
  };
}
