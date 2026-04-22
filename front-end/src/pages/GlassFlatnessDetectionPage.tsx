import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ImageUploader } from "@/utils/ImageUploader";
import { DetectionResult } from "@/components/DetectionResult";
import type { DetectionResultData } from "@/types/detection";
import { Loader2, Ruler } from "lucide-react";
import { useImageSlots } from "@/hooks/useImageSlots";
import { detectGlassFlatness, type FlatnessFieldName } from "@/services/detectApi";
import { PageHeader } from "@/components/layout/PageHeader";
import { InstructionList } from "@/components/layout/InstructionList";
import { useAuth } from "@/hooks/useAuth";

export function GlassFlatnessDetectionPage() {
  const maxCount = 4; // 可以上传的最大图片数量
  const fieldNames = ["left_env", "left_mix", "right_env", "right_mix"] as const satisfies readonly FlatnessFieldName[];
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<DetectionResultData | null>(null);
  const { auth } = useAuth();
  const { files, previewUrls, currentIndex, setCurrentIndex, setFileAt, removeAt, isComplete, filledCount } = useImageSlots(maxCount);

  // 调用后端逻辑
  const handleDetect = async () => {
    if (!isComplete) return;

    setIsUploading(true);
    try {
      const payload: Partial<Record<FlatnessFieldName, File>> = {};
      fieldNames.forEach((field, idx) => {
        const file = files[idx];
        if (file) payload[field] = file;
      });
      const user = auth?.user;
      if (!user) {
        setResult({
          status: "error",
          title: "用户未登录",
          description: "请先登录后再进行操作。",
        });
        return;
      }
      setResult(await detectGlassFlatness(user.id, payload));
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
          icon={<Ruler className="w-8 h-8 text-white" />}
          title="玻璃幕墙平整度检测"
          description="按左右环境图与投影图完成平整度分析，保留结果图与 3D 点云地图用于复核。"
          iconBgClassName="bg-gradient-to-br from-blue-500 to-indigo-600 shadow-blue-500/30"
        />

        <Card className="workbench-panel">
          <div className="workbench-grid">
            <div>
              <div className="panel-title">
                <h3>上传四组图像</h3>
                <span>{filledCount}/{maxCount} 已就绪</span>
              </div>
              <ImageUploader
                files={files}
                previewUrls={previewUrls}
                currentIndex={currentIndex}
                onCurrentIndexChange={setCurrentIndex}
                disabled={isUploading}
                slotTips={["请上传左侧环境图", "请上传左侧投影图", "请上传右侧环境图", "请上传右侧投影图"]}
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
                <span>FLATNESS MAP</span>
              </div>
              <Card className="glass-panel action-panel">
                <div className="space-y-6">
                  <div>
                    <p className="panel-title">现场拍摄要求</p>
                    <InstructionList
                      colorClassName="text-blue-400"
                      items={[
                        "拍摄角度尽量垂直于幕墙表面",
                        "保持3-5米的适当拍摄距离",
                        "确保整体幕墙结构在画面中",
                        "光线均匀，避免强烈阴影",
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
            <div className="section-head">
              <div>
                <h2>检测结果</h2>
                <p>平整度结果支持结果图与点云交互复核。</p>
              </div>
            </div>
            <DetectionResult result={result} />
          </div>
        )}
      </div>
    </div>
  );
}
