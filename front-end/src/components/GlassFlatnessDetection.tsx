import { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { ImageUploader } from "./ImageUploader";
import { DetectionResult } from "./DetectionResult";
import type { DetectionResultData } from "./DetectionResult";
import { Loader2, Ruler } from "lucide-react";

export function GlassFlatnessDetection() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResultData | null>(null);

  // 选择图片
  const handleImageSelect = (file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
    setResult(null);
  };

  // 删除图片
  const handleImageRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setImageFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  // 调用后端逻辑
  const handleDetect = async () => {
    if (!imageFile) return;

    setIsDetecting(true);
    
    // 模拟 API 调用
    // TODO: 待添加后端逻辑
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 模拟检测结果
    // TODO: 待添加读取结果
    const mockResults: DetectionResultData[] = [
      {
        status: "success",
        title: "平整度良好",
        description: "玻璃幕墙平整度符合标准要求，表面平整无明显凸起或凹陷。",
        details: [
          { label: "平整度偏差", value: "±0.8mm" },
          { label: "标准要求", value: "±3mm" },
          { label: "检测点数", value: "25个" },
          { label: "合格率", value: "100%" }
        ]
      },
      {
        status: "error",
        title: "平整度不合格",
        description: "检测到明显的平整度问题，部分区域偏差超出标准范围。",
        details: [
          { label: "最大偏差", value: "±5.2mm" },
          { label: "标准要求", value: "±3mm" },
          { label: "不合格区域", value: "3处" },
          { label: "合格率", value: "68%" }
        ]
      },
      {
        status: "warning",
        title: "平整度临界",
        description: "部分区域平整度接近标准上限，建议关注并进行定期检测。",
        details: [
          { label: "平整度偏差", value: "±2.8mm" },
          { label: "标准要求", value: "±3mm" },
          { label: "临界区域", value: "2处" },
          { label: "合格率", value: "92%" }
        ]
      }
    ];

    const randomResult = mockResults[Math.floor(Math.random() * mockResults.length)];
    setResult(randomResult);
    setIsDetecting(false);
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
                onImageSelect={handleImageSelect}
                onImageRemove={handleImageRemove}
                previewUrl={previewUrl}
                disabled={isDetecting}
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
                    disabled={!imageFile || isDetecting}
                    className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white border-0 shadow-lg shadow-blue-500/30"
                    size="lg"
                  >
                    {isDetecting ? (
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