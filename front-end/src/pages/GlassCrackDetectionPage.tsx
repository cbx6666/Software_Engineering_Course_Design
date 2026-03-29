import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ImageUploader } from "@/utils/ImageUploader";
import { DetectionResult } from "@/components/DetectionResult";
import type { DetectionResultData } from "@/types/detection";
import { Loader2, ScanSearch } from "lucide-react";
import { useImageSlots } from "@/hooks/useImageSlots";
import { detectGlassCrack } from "@/services/detectApi";
import { PageHeader } from "@/components/layout/PageHeader";
import { InstructionList } from "@/components/layout/InstructionList";
import { useAuth } from "@/hooks/useAuth";

export function GlassCrackDetectionPage() {
  const [result, setResult] = useState<DetectionResultData | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { auth } = useAuth();
  const maxCount = 1; // 最大上传图片数，可根据需求调整
  const { files, previewUrls, currentIndex, setCurrentIndex, setFileAt, removeAt, isComplete, filledCount } = useImageSlots(maxCount);

  /** 调用后端逻辑，上传所有图片并返回统一结果 */
  const handleDetect = async () => {
    if (!isComplete) return;
    const uploadFiles = files.filter(Boolean) as File[];

    setIsUploading(true);
    try {
      const user = auth?.user;
      if (!user) {
        setResult({
          status: "error",
          title: "用户未登录",
          description: "请先登录后再进行操作。",
        });
        return;
      }
      setResult(await detectGlassCrack(user.id, uploadFiles));
    } catch {
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
        <PageHeader
          icon={<ScanSearch className="w-8 h-8 text-white" />}
          title="玻璃破裂检测"
          description="上传玻璃图片，系统将自动检测是否存在裂纹或破损"
          iconBgClassName="bg-gradient-to-br from-cyan-500 to-blue-600 shadow-cyan-500/30"
        />

        {/* Main Card */}
        <Card className="border-0 bg-white/10 backdrop-blur-xl p-8 shadow-2xl">
          <div className="grid md:grid-cols-2 gap-8">
            {/* 图片上传区 */}
            <div>
              <h3 className="text-white mb-4">上传图片</h3>
              <ImageUploader
                currentIndex={currentIndex}
                disabled={isUploading}
                onCurrentIndexChange={setCurrentIndex}
                files={files}
                previewUrls={previewUrls}
                onSelectFile={(idx, file) => {
                  setFileAt(idx, file);
                  setResult(null);
                }}
                onRemove={(idx) => {
                  removeAt(idx);
                  setResult(null);
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
                    <InstructionList
                      colorClassName="text-cyan-400"
                      items={[
                        "支持常见图片格式（JPG、PNG、WEBP）",
                        "建议图片清晰度较高，分辨率≥1080p",
                        "光线充足效果更佳，避免强烈反光",
                        "确保检测区域无遮挡",
                      ]}
                    />
                  </div>

                  {filledCount < maxCount && (
                    <p className="text-red-400 text-sm mt-2">
                      请上传 {maxCount} 张图片后才能开始检测
                    </p>
                  )}

                  <Button
                    onClick={handleDetect}
                    disabled={!isComplete || isUploading}
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
