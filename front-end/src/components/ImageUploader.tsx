// src/utils/imageOperation.ts
import { useState } from "react";
import type { DetectionResultData } from "../components/DetectionResult";
import axios from "axios";

export function useImageUpload(apiUrl: string) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // 选择图片
  const selectImage = (file: File) => {
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // 移除图片
  const removeImage = () => {
    setImageFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  // 上传图片到后端
  const uploadImage = async (): Promise<DetectionResultData> => {
    if (!imageFile) throw new Error("没有选择图片");

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);

      const response = await axios.post<DetectionResultData>(apiUrl, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      return response.data;
    } catch (err: any) {
      // 可以根据 Axios 错误对象详细处理
      if (err.response) {
        throw new Error(`上传失败, HTTPS ${err.response.status}`);
      } else if (err.request) {
        throw new Error("上传失败，未收到服务器响应");
      } else {
        throw new Error("上传失败：" + err.message);
      }
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
