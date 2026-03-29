import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { DetectionResult } from "@/components/detection/DetectionResult";
import { getHistoryItemById, type HistoryItem } from "@/services/historyApi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";

export function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [historyItem, setHistoryItem] = useState<HistoryItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080';

  useEffect(() => {
    if (id) {
      setLoading(true);
      getHistoryItemById(id)
        .then(item => {
          setHistoryItem(item);
        })
        .catch(() => {
          setHistoryItem(null);
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [id]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      </div>
    );
  }

  if (!historyItem) {
    return <div className="text-center text-white mt-10">未找到该历史记录</div>;
  }

  return (
    <div className="container mx-auto p-4 md:p-6 lg:p-8 max-w-6xl">
      <PageHeader
        title="检测详情"
        description={`查看 ${historyItem.title} 的详细信息`}
        icon={<ArrowLeft className="w-8 h-8 text-white cursor-pointer" onClick={() => navigate(-1)} />}
        iconBgClassName="bg-gray-500"
      />
      <div className="grid gap-6 mt-6">
        <Card className="border-0 bg-white/10 backdrop-blur-xl shadow-lg">
          <CardHeader>
            <CardTitle className="text-white">基本信息</CardTitle>
          </CardHeader>
          <CardContent className="text-slate-300">
            <p><strong>标题:</strong> {historyItem.title}</p>
            <p><strong>日期:</strong> {new Date(historyItem.date).toLocaleString('zh-CN')}</p>
            <p><strong>描述:</strong> {historyItem.description}</p>
          </CardContent>
        </Card>

        {/* 待检测图片展示区 */}
        {historyItem.originalImages && historyItem.originalImages.length > 0 && (
          <Card className="border-0 bg-white/10 backdrop-blur-xl shadow-lg">
            <CardHeader>
              <CardTitle className="text-white">待检测图片</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-4">
              <div className="relative w-full max-w-lg flex justify-center items-center">
                {historyItem.originalImages.length > 1 && (
                  <button
                    onClick={() => {
                    if (historyItem.originalImages && historyItem.originalImages.length > 1) {
                      setCurrentImageIndex((prev) =>
                        prev === 0 ? historyItem.originalImages!.length - 1 : prev - 1
                      );
                    }
                  }}
                    className="absolute left-0 z-10 p-2 bg-black/50 rounded-full text-white hover:bg-black/75 transition-colors"
                    aria-label="Previous Image"
                  >
                    <ChevronLeft className="w-6 h-6" />
                  </button>
                )}
                <img
                  src={`${API_BASE_URL}${historyItem.originalImages[currentImageIndex]}`}
                  alt={`检测图片 ${currentImageIndex + 1}`}
                  className="w-full h-auto max-h-[60vh] object-contain rounded-md"
                />
                {historyItem.originalImages.length > 1 && (
                  <button
                    onClick={() => {
                    if (historyItem.originalImages && historyItem.originalImages.length > 1) {
                      setCurrentImageIndex((prev) =>
                        prev === historyItem.originalImages!.length - 1 ? 0 : prev + 1
                      );
                    }
                  }}
                    className="absolute right-0 z-10 p-2 bg-black/50 rounded-full text-white hover:bg-black/75 transition-colors"
                    aria-label="Next Image"
                  >
                    <ChevronRight className="w-6 h-6" />
                  </button>
                )}
              </div>
              {historyItem.originalImages.length > 1 && (
                <p className="text-sm text-slate-400">
                  {currentImageIndex + 1} / {historyItem.originalImages.length}
                </p>
              )}
            </CardContent>
          </Card>
        )}

        <DetectionResult result={historyItem as HistoryItem} />
      </div>
    </div>
  );
}
