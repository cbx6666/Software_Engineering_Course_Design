import { useMemo, useRef, useState, type ChangeEvent, type DragEvent, type MouseEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, X, ChevronLeft, ChevronRight } from "lucide-react";

interface ImageUploaderProps {
  files: (File | null)[];
  previewUrls: (string | null)[];
  currentIndex?: number;
  onCurrentIndexChange?: (index: number) => void;
  disabled?: boolean;
  slotTips?: string[];
  accept?: string;
  onSelectFile: (index: number, file: File) => void;
  onRemove: (index: number) => void;
}

export function ImageUploader({
  files,
  previewUrls,
  currentIndex: currentIndexProp,
  onCurrentIndexChange,
  disabled = false,
  slotTips,
  accept = "image/*",
  onSelectFile,
  onRemove,
}: ImageUploaderProps) {
  const maxCount = files.length;

  const [uncontrolledIndex, setUncontrolledIndex] = useState(0);
  const currentIndex = currentIndexProp ?? uncontrolledIndex;
  const setCurrentIndex = onCurrentIndexChange ?? setUncontrolledIndex;

  const fileInputRef = useRef<HTMLInputElement>(null);

  const filledCount = useMemo(() => files.filter(Boolean).length, [files]);

  // 选择文件（真正的数据写入由外部完成）
  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    onSelectFile(currentIndex, file);

    // 自动切换到下一个空槽位（不改变顺序）
    const nextEmpty = files.findIndex((f) => !f);
    if (nextEmpty >= 0) setCurrentIndex(nextEmpty);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  // 左右切换槽位
  const handlePrev = () => {
    setCurrentIndex((prev) => (prev - 1 + maxCount) % maxCount);
  };
  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % maxCount);
  };

  // 拖拽上传
  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
  };
  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const tips = slotTips ?? [
    "请上传左侧环境图",
    "请上传左侧投影图",
    "请上传右侧环境图",
    "请上传右侧投影图",
  ];

  return (
    <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-white/10 relative">
      {/* 上传槽位预览 */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
          }`}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {files[currentIndex] && previewUrls[currentIndex] ? (
          <div className="relative">
            <img
              src={previewUrls[currentIndex]!}
              alt={`Preview ${currentIndex}`}
              className="w-full h-64 object-contain rounded-lg"
            />
            <Button
              variant="destructive"
              size="icon"
              className="absolute top-2 right-2 z-20 shadow-lg"
              onClick={(e: MouseEvent) => {
                e.stopPropagation();
                onRemove(currentIndex);
              }}
              disabled={disabled}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-sm flex items-center justify-center border border-cyan-500/30">
              <Upload className="w-8 h-8 text-cyan-400" />
            </div>
            <p className="text-white mb-1">点击上传或拖拽图片到此处</p>
            <p className="text-slate-400">支持 JPG、PNG、WEBP 格式</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
          disabled={disabled}
        />
      </div>


      {/* 槽位索引 */}
      {maxCount > 1 && (
        <div>
          <p className="text-red-400 text-sm mt-2">
            {files[currentIndex] ? "" : (tips[currentIndex] ?? "")}
          </p>

          <div className="text-white mt-2 text-sm text-center">
            {currentIndex + 1}/{maxCount}
          </div>
        </div>
      )}

      {/* 卡片下方的左右箭头 */}
      {maxCount > 1 && (
        <div className="flex justify-center items-center gap-24">
          <Button
            className="bg-transparent hover:scale-120 cursor-pointer transition"
            onClick={handlePrev}
          >
            <ChevronLeft className="w-10 h-10 text-white" />
          </Button>
          <Button
            className="bg-transparent hover:scale-120 cursor-pointer transition"
            onClick={handleNext}
          >
            <ChevronRight className="w-10 h-10 text-white" />
          </Button>
        </div>
      )}

      {/* 辅助信息 */}
      {maxCount > 1 && (
        <div className="mt-4 text-center text-xs text-slate-400">
          已上传 {filledCount}/{maxCount}
        </div>
      )}
    </Card>
  );
}