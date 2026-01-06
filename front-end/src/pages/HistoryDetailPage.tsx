import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { DetectionResult } from "@/components/detection/DetectionResult";
import { getHistoryItemById, type HistoryItem } from "@/services/historyApi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Loader2 } from "lucide-react";

export function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [historyItem, setHistoryItem] = useState<HistoryItem | null>(null);
  const [loading, setLoading] = useState(true);

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
    <div className="container mx-auto p-4 md:p-6 lg:p-8">
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

        <DetectionResult result={historyItem as HistoryItem} />
      </div>
    </div>
  );
}
