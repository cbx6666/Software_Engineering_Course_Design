import { useEffect, useMemo, useRef, useState } from "react";

type MaybeFile = File | null;
type MaybeUrl = string | null;

function createEmpty<T>(count: number, value: T) {
  return Array.from({ length: count }, () => value);
}

export function useImageSlots(maxCount: number) {
  const [files, setFiles] = useState<MaybeFile[]>(() => createEmpty(maxCount, null));
  const [previewUrls, setPreviewUrls] = useState<MaybeUrl[]>(() => createEmpty(maxCount, null));
  const [currentIndex, setCurrentIndex] = useState(0);

  const urlsRef = useRef<(string | null)[]>(createEmpty(maxCount, null));

  // maxCount 变化时，重置为新长度，避免越界
  useEffect(() => {
    // revoke old urls
    urlsRef.current.forEach((u) => u && URL.revokeObjectURL(u));
    urlsRef.current = createEmpty(maxCount, null);
    setFiles(createEmpty(maxCount, null));
    setPreviewUrls(createEmpty(maxCount, null));
    setCurrentIndex(0);
  }, [maxCount]);

  // 组件卸载时 revoke
  useEffect(() => {
    return () => {
      urlsRef.current.forEach((u) => u && URL.revokeObjectURL(u));
    };
  }, []);

  const filledCount = useMemo(() => files.filter(Boolean).length, [files]);
  const isComplete = filledCount === maxCount;

  const setFileAt = (index: number, file: File | null) => {
    setFiles((prev) => {
      const next = [...prev];
      next[index] = file;
      return next;
    });

    setPreviewUrls((prev) => {
      const next = [...prev];
      const oldUrl = urlsRef.current[index];
      if (oldUrl) URL.revokeObjectURL(oldUrl);

      const newUrl = file ? URL.createObjectURL(file) : null;
      urlsRef.current[index] = newUrl;
      next[index] = newUrl;
      return next;
    });
  };

  const removeAt = (index: number) => setFileAt(index, null);

  const reset = () => {
    urlsRef.current.forEach((u) => u && URL.revokeObjectURL(u));
    urlsRef.current = createEmpty(maxCount, null);
    setFiles(createEmpty(maxCount, null));
    setPreviewUrls(createEmpty(maxCount, null));
    setCurrentIndex(0);
  };

  return {
    files,
    previewUrls,
    currentIndex,
    setCurrentIndex,
    setFileAt,
    removeAt,
    reset,
    filledCount,
    isComplete,
  };
}


