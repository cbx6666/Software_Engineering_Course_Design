import { useState, useRef } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Upload, X, Image as ImageIcon } from "lucide-react";

interface ImageUploaderProps {
  onImageSelect: (file: File, previewUrl: string) => void;
  onImageRemove: () => void;
  previewUrl: string | null;
  disabled?: boolean;
}

export function ImageUploader({
  onImageSelect,
  onImageRemove,
  previewUrl,
  disabled = false
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      onImageSelect(file, url);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (!disabled) {
      const file = e.dataTransfer.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    }
  };

  const handleRemove = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onImageRemove();
  };

  return (
    <Card className="p-6 bg-slate-900/30 backdrop-blur-md border-white/10">
      {!previewUrl ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            isDragging
              ? "border-cyan-400 bg-cyan-500/10"
              : "border-white/20 hover:border-cyan-400/50 hover:bg-white/5"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <div className="flex flex-col items-center gap-3">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-sm flex items-center justify-center border border-cyan-500/30">
              <Upload className="w-8 h-8 text-cyan-400" />
            </div>
            <div>
              <p className="text-white mb-1">
                点击上传或拖拽图片到此处
              </p>
              <p className="text-slate-400">
                支持 JPG、PNG、WEBP 格式
              </p>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            disabled={disabled}
          />
        </div>
      ) : (
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <img
            src={previewUrl}
            alt="Preview"
            className="w-full h-auto rounded-lg relative z-10"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-2 right-2 z-20 shadow-lg"
            onClick={handleRemove}
            disabled={disabled}
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      )}
    </Card>
  );
}