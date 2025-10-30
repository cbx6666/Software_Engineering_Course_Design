import { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { ImageUploader } from "./ImageUploader";
import { DetectionResult, DetectionResultData } from "./DetectionResult";
import { Loader2, ScanSearch } from "lucide-react";

export function GlassCrackDetection() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResultData | null>(null);

  const handleImageSelect = (file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const handleImageRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setImageFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  const handleDetect = async () => {
    if (!imageFile) return;

    setIsDetecting(true);
    
    // 模拟 API 调用
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 模拟检测结果
    const mockResults: DetectionResultData[] = [
      {
        status: "success",
        title: "检测完成",
        description: "未检测到玻璃破裂，玻璃完好无损。",
        details: [
          { label: "置信度", value: "98.5%" },
          { label: "检测区域", value: "全覆盖" },
          { label: "处理时间", value: "1.2秒" }
        ]
      },
      {
        status: "error",
        title: "检测到破裂",
        description: "玻璃存在明显裂纹，建议立即更换以确保安全。",
        details: [
          { label: "裂纹数量", value: "3处" },
          { label: "裂纹长度", value: "15-25cm" },
          { label: "危险等级", value: "高" },
          { label: "置信度", value: "95.2%" }
        ]
      },
      {
        status: "warning",
        title: "疑似轻微损伤",
        description: "检测到可能的细微裂纹或划痕，建议进一步检查。",
        details: [
          { label: "可疑区域", value: "2处" },
          { label: "置信度", value: "72.8%" },
          { label: "建议", value: "人工复核" }
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
                  <Button
                    onClick={handleDetect}
                    disabled={!imageFile || isDetecting}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0 shadow-lg shadow-cyan-500/30"
                    size="lg"
                  >
                    {isDetecting ? (
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