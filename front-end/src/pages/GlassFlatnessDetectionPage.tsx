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
    <div className="min-h-screen p-8 md:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Page Header */}
        <PageHeader
          icon={<Ruler className="w-8 h-8 text-white" />}
          title="玻璃幕墙平整度检测"
          description="上传幕墙图片，系统将分析玻璃表面的平整度"
          iconBgClassName="bg-gradient-to-br from-blue-500 to-indigo-600 shadow-blue-500/30"
        />

        {/* Main Card */}
        <Card className="border-0 bg-white/10 backdrop-blur-xl p-8 shadow-2xl">
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-white mb-4">上传图片</h3>
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
              <h3 className="text-white mb-4">操作面板</h3>
              <Card className="p-6 bg-slate-900/50 backdrop-blur-md border-white/10">
                <div className="space-y-6">
                  <div>
                    <p className="text-white mb-3">检测说明</p>
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
                    <p className="text-red-400 text-sm mt-2">
                      请上传 {maxCount} 张图片后才能开始检测
                    </p>
                  )}

                  <Button
                    onClick={handleDetect}
                    disabled={!isComplete || isUploading}
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
