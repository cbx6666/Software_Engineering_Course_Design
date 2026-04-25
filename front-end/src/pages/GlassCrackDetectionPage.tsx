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
    <div className="lab-page">
      <div className="lab-container">
        <PageHeader
          icon={<ScanSearch className="w-8 h-8 text-white" />}
          title="玻璃破裂检测"
          description="上传玻璃图片，系统将检测裂纹、崩边与明显破损，并返回可复核的检测结果。"
          iconBgClassName="bg-gradient-to-br from-cyan-500 to-blue-600 shadow-cyan-500/30"
        />

        <Card className="workbench-panel">
          <div className="workbench-grid">
            <div>
              <div className="panel-title">
                <h3>上传检测图像</h3>
                <span>{filledCount}/{maxCount} 已就绪</span>
              </div>
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

            <div>
              <div className="panel-title">
                <h3>检测作业面板</h3>
                <span>CRACK SCAN</span>
              </div>
              <Card className="glass-panel action-panel">
                <div className="space-y-6">
                  <div>
                    <p className="panel-title">图像采集建议</p>
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
                    <p className="upload-warning">
                      请上传 {maxCount} 张图片后才能开始检测
                    </p>
                  )}

                  <Button
                    onClick={handleDetect}
                    disabled={!isComplete || isUploading}
                    className="lab-primary-button"
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

        {result && (
          <div>
            <div className="section-head">
              <div>
                <h2>检测结果</h2>
                <p>结果来自当前上传图像，可继续替换图片重新检测。</p>
              </div>
            </div>
            <DetectionResult result={result} />
          </div>
        )}
      </div>
    </div>
  );
}
