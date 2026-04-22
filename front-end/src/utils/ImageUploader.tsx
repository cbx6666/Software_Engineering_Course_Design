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
    <Card className="glass-panel uploader-card">
      <div
        className={`drop-zone ${disabled ? "drop-zone--disabled" : ""}`}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {files[currentIndex] && previewUrls[currentIndex] ? (
          <div className="drop-zone__preview">
            <img
              src={previewUrls[currentIndex]!}
              alt={`Preview ${currentIndex}`}
            />
            <Button
              variant="destructive"
              size="icon"
              className="uploader-remove"
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
          <div className="drop-zone__empty">
            <div className="drop-zone__icon">
              <Upload />
            </div>
            <strong>点击上传或拖拽图片到此处</strong>
            <span>支持 JPG、PNG、WEBP 格式，建议使用清晰原图</span>
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


      {maxCount > 1 && (
        <div className="slot-strip">
          <Button
            className="slot-nav"
            onClick={handlePrev}
            type="button"
          >
            <ChevronLeft className="w-5 h-5" />
          </Button>

          <div className="slot-status">
            <p className="slot-status__tip">
              {files[currentIndex] ? "当前槽位已完成" : (tips[currentIndex] ?? "")}
            </p>
            <div className="slot-progress">
              <span style={{ width: `${(filledCount / maxCount) * 100}%` }} />
            </div>
            <div className="slot-status__count">
              槽位 {currentIndex + 1}/{maxCount} · 已上传 {filledCount}/{maxCount}
            </div>
          </div>

          <Button
            className="slot-nav"
            onClick={handleNext}
            type="button"
          >
            <ChevronRight className="w-5 h-5" />
          </Button>
        </div>
      )}

      {maxCount === 1 && (
        <div className="slot-status mt-4">
          <div className="slot-progress">
            <span style={{ width: filledCount ? "100%" : "0%" }} />
          </div>
        </div>
      )}
    </Card>
  );
}
