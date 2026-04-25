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
      <div className="lab-page flex justify-center items-center">
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      </div>
    );
  }

  if (!historyItem) {
    return <div className="lab-page history-empty">未找到该历史记录</div>;
  }

  return (
    <div className="lab-page">
      <div className="lab-container">
      <PageHeader
        title="检测详情"
        description={`查看 ${historyItem.title} 的原始图像、检测结论与复核指标。`}
        icon={<ArrowLeft className="w-8 h-8 text-white cursor-pointer" onClick={() => navigate(-1)} />}
        iconBgClassName="bg-gray-500"
      />
      <div className="detail-layout">
        <Card className="history-detail-panel">
          <CardHeader>
            <CardTitle className="panel-title">基本信息</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="info-grid">
              <div className="info-row"><strong>标题</strong><span>{historyItem.title}</span></div>
              <div className="info-row"><strong>日期</strong><span>{new Date(historyItem.date).toLocaleString('zh-CN')}</span></div>
              <div className="info-row"><strong>描述</strong><span>{historyItem.description}</span></div>
            </div>
          </CardContent>
        </Card>

        {historyItem.originalImages && historyItem.originalImages.length > 0 && (
          <Card className="history-detail-panel">
            <CardHeader>
              <CardTitle className="panel-title">待检测图片</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="carousel-frame">
                {historyItem.originalImages.length > 1 && (
                  <button
                    onClick={() => {
                    if (historyItem.originalImages && historyItem.originalImages.length > 1) {
                      setCurrentImageIndex((prev) =>
                        prev === 0 ? historyItem.originalImages!.length - 1 : prev - 1
                      );
                    }
                  }}
                    className="carousel-button carousel-button--left"
                    aria-label="Previous Image"
                  >
                    <ChevronLeft className="w-6 h-6" />
                  </button>
                )}
                <img
                  src={`${API_BASE_URL}${historyItem.originalImages[currentImageIndex]}`}
                  alt={`检测图片 ${currentImageIndex + 1}`}
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
                    className="carousel-button carousel-button--right"
                    aria-label="Next Image"
                  >
                    <ChevronRight className="w-6 h-6" />
                  </button>
                )}
              </div>
              {historyItem.originalImages.length > 1 && (
                <p className="carousel-count">
                  {currentImageIndex + 1} / {historyItem.originalImages.length}
                </p>
              )}
            </CardContent>
          </Card>
        )}

        <DetectionResult result={historyItem as HistoryItem} />
      </div>
      </div>
    </div>
  );
}
