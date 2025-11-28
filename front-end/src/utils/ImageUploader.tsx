import { useState, useRef } from "react";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Upload, X, ChevronLeft, ChevronRight } from "lucide-react";

interface ImageUploaderProps {
  maxCount?: number; // 最大上传数量
  files: (File | null)[];
  previewUrls: (string | null)[];
  currentIndex?: number; // 可选，当前聚焦槽位
  disabled?: boolean;
  onImagesChange: (newFiles: (File | null)[], newPreviews: (string | null)[]) => void;
}

export function ImageUploader({
  maxCount = 1,
  onImagesChange,
  disabled = false
}: ImageUploaderProps) {
  const [files, setFiles] = useState<(File | null)[]>(Array(maxCount).fill(null));
  const [previewUrls, setPreviewUrls] = useState<(string | null)[]>(Array(maxCount).fill(null));
  const [currentIndex, setCurrentIndex] = useState(0); // 当前聚焦槽位
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 选择文件
  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) return;

    const newFiles = [...files];
    const newPreviews = [...previewUrls];

    newFiles[currentIndex] = file;
    if (newPreviews[currentIndex]) URL.revokeObjectURL(newPreviews[currentIndex]);
    newPreviews[currentIndex] = URL.createObjectURL(file);

    setFiles(newFiles);
    setPreviewUrls(newPreviews);

    // 回调父组件
    onImagesChange(newFiles, newPreviews);

    // 自动切换到下一个空槽位
    const nextEmpty = newFiles.findIndex(f => !f);
    if (nextEmpty >= 0) setCurrentIndex(nextEmpty);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  // 删除图片
  const handleRemove = (index: number) => {
    const newFiles: (File | null)[] = [];
    const newPreviews: (string | null)[] = [];

    // 遍历原数组，把非空的图片都先加入
    for (let i = 0; i < files.length; i++) {
      if (i !== index && files[i]) {
        newFiles.push(files[i]);
        newPreviews.push(previewUrls[i]);
      }
    }

    // 填充剩余槽位为空
    while (newFiles.length < maxCount) newFiles.push(null);
    while (newPreviews.length < maxCount) newPreviews.push(null);

    setFiles(newFiles);
    setPreviewUrls(newPreviews);
    onImagesChange(newFiles, newPreviews);

    // 聚焦第一个空槽位
    const firstEmptyIndex = newFiles.findIndex(f => f === null);
    setCurrentIndex(firstEmptyIndex >= 0 ? firstEmptyIndex : 0);
  };

  // 左右切换槽位
  const handlePrev = () => {
    setCurrentIndex((prev) => (prev - 1 + maxCount) % maxCount);
  };
  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % maxCount);
  };

  // 拖拽上传
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

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
              onClick={(e: any) => {
                e.stopPropagation();
                handleRemove(currentIndex);
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
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          disabled={disabled}
        />
      </div>

      {/* 槽位索引 */}
      {maxCount > 1 && (
        <div className="text-white mt-2 text-sm text-center">
          {currentIndex + 1}/{maxCount}
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
    </Card>
  );
}