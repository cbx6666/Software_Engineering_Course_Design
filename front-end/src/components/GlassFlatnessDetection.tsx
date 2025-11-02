import { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { ImageUploader } from "./ImageUploader";
import { DetectionResult } from "./DetectionResult";
import type { DetectionResultData } from "./DetectionResult";
import { Loader2, Ruler } from "lucide-react";

import { useImageUpload } from "../utils/imageOperation";

export function GlassFlatnessDetection() {
  const [result, setResult] = useState<DetectionResultData | null>(null);
  const { imageFile, previewUrl, isUploading, selectImage, removeImage, uploadImage } =
    useImageUpload("xxx"); // TODO: 待添加后端接口

  // 调用后端逻辑
  const handleDetect = async () => {
    if (!imageFile) return;

    try {
      const result = await uploadImage();
      setResult(result);
    } catch (err) {
      console.error("检测失败:", err);
      setResult({
        status: "error",
        title: "上传失败",
        description: "图片上传或检测过程出现错误，请稍后再试。",
        details: [],
      });
    };
  };

  return (
    <div className="min-h-screen p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Page Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 mb-4 shadow-xl shadow-blue-500/30">
            <Ruler className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-white mb-3">玻璃幕墙平整度检测</h1>
          <p className="text-slate-300 max-w-2xl mx-auto">
            上传幕墙图片，系统将分析玻璃表面的平整度
          </p>
        </div>

        {/* Main Card */}
        <Card className="border-0 bg-white/10 backdrop-blur-xl p-8 shadow-2xl">
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-white mb-4">上传图片</h3>
              <ImageUploader
                onImageSelect={selectImage}
                onImageRemove={removeImage}
                previewUrl={previewUrl}
                disabled={isUploading}
              />
            </div>

            <div>
              <h3 className="text-white mb-4">操作面板</h3>
              <Card className="p-6 bg-slate-900/50 backdrop-blur-md border-white/10">
                <div className="space-y-6">
                  <div>
                    <p className="text-white mb-3">检测说明</p>
                    <ul className="text-slate-300 space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>拍摄角度尽量垂直于幕墙表面</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>保持3-5米的适当拍摄距离</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>确保整体幕墙结构在画面中</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>光线均匀，避免强烈阴影</span>
                      </li>
                    </ul>
                  </div>
                  <Button
                    onClick={handleDetect}
                    disabled={!imageFile || isUploading}
                    className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white border-0 shadow-lg shadow-blue-500/30"
                    size="lg"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        AI分析中...
                      </>
                    ) : (
                      <>
                        <Ruler className="w-4 h-4 mr-2" />
                        开始检测
                      </>
                    )}
                  </Button>
                </div>
              </Card>
            </div>
          </div>
        </Card>

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