import { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { ImageUploader } from "../utils/ImageUploader";
import { DetectionResult } from "./DetectionResult";
import type { DetectionResultData } from "./DetectionResult";
import { Loader2, ScanSearch } from "lucide-react";
import axios from "axios";

export function GlassCrackDetection() {
  const [files, setFiles] = useState<(File | null)[]>([]);
  const [previewUrls, setPreviewUrls] = useState<(string | null)[]>([]);
  const [result, setResult] = useState<DetectionResultData | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0); // 当前聚焦槽位

  const backendUrl = import.meta.env.VITE_BACKEND_URL;
  const maxCount = 1; // 最大上传图片数，可根据需求调整

  /** 调用后端逻辑，上传所有图片并返回统一结果 */
  const handleDetect = async () => {
    const uploadFiles = files.filter((f) => f !== null) as File[];
    if (uploadFiles.length === 0) return;

    setIsUploading(true);
    try {
      const formData = new FormData();
      uploadFiles.forEach((file) => formData.append("images", file)); // 后端接收 images[]
      const response = await axios.post(`${backendUrl}/api/detect/glass-crack`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        withCredentials: true,
      });

      setResult(response.data); // 统一检测结果
    } catch (err: any) {
      setResult({
        status: "error",
        title: "上传失败",
        description: "图片上传或检测过程出现错误，请稍后再试。",
        details: [],
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Page Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 mb-4 shadow-xl shadow-cyan-500/30">
            <ScanSearch className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-white mb-3">玻璃破裂检测</h1>
          <p className="text-slate-300 max-w-2xl mx-auto">
            上传玻璃图片，系统将自动检测是否存在裂纹或破损
          </p>
        </div>

        {/* Main Card */}
        <Card className="border-0 bg-white/10 backdrop-blur-xl p-8 shadow-2xl">
          <div className="grid md:grid-cols-2 gap-8">
            {/* 图片上传区 */}
            <div>
              <h3 className="text-white mb-4">上传图片</h3>
              <ImageUploader
                maxCount={maxCount}
                currentIndex={currentIndex}
                files={files}
                previewUrls={previewUrls}
                disabled={isUploading}
                onImagesChange={(newFiles, newPreviews) => {
                  setFiles(newFiles);
                  setPreviewUrls(newPreviews);
                  setResult(null); // 删除或上传时清空检测结果
                  // 聚焦第一个空槽位
                  const firstEmptyIndex = newFiles.findIndex((f) => f === null);
                  setCurrentIndex(firstEmptyIndex >= 0 ? firstEmptyIndex : 0);
                }}
              />
            </div>

            {/* 操作面板 */}
            <div>
              <h3 className="text-white mb-4">操作面板</h3>
              <Card className="p-6 bg-slate-900/50 backdrop-blur-md border-white/10">
                <div className="space-y-6">
                  <div>
                    <p className="text-white mb-3">检测说明</p>
                    <ul className="text-slate-300 space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>支持常见图片格式（JPG、PNG、WEBP）</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>建议图片清晰度较高，分辨率≥1080p</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>光线充足效果更佳，避免强烈反光</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>确保检测区域无遮挡</span>
                      </li>
                    </ul>
                  </div>

                  {files.filter(f => f !== null).length < maxCount && (
                    <p className="text-red-400 text-sm mt-2">
                      请上传 {maxCount} 张图片后才能开始检测
                    </p>
                  )}

                  <Button
                    onClick={handleDetect}
                    disabled={files.filter((f) => f !== null).length != maxCount || isUploading}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0 shadow-lg shadow-cyan-500/30"
                    size="lg"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        AI检测中...
                      </>
                    ) : (
                      <>
                        <ScanSearch className="w-4 h-4 mr-2" />
                        开始检测
                      </>
                    )}
                  </Button>
                </div>
              </Card>
            </div>
          </div>
        </Card>

        {/* 检测结果 */}
        {result && (
          <div>
            <h3 className="text-white mb-4">检测结果</h3>
            <DetectionResult result={result} />
          </div>
        )}
      </div>
    </div>
  );
}
