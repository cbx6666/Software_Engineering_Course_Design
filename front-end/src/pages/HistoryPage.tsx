import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { HistoryItemCard } from "@/components/history/HistoryItemCard";
import { getHistory, type HistoryItem } from "@/services/historyApi";
import { useAuth } from "@/hooks/useAuth";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/CustomTabs";
import { History as HistoryIcon, Loader2 } from "lucide-react";
import { buildRoutePath } from "@/routes";

export function HistoryPage() {
  const navigate = useNavigate();
  const { auth } = useAuth();
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const user = auth?.user;
    if (user) {
      setLoading(true);
      getHistory(user.id)
        .then(items => {
          setHistoryItems(items);
        })
        .catch(error => {
          console.error("Failed to fetch history:", error);
          setHistoryItems([]); // On error, show an empty list
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      // If not logged in, ensure the list is empty and stop loading
      setHistoryItems([]);
      setLoading(false);
    }
  }, [auth]);

  const handleItemClick = (id: string) => {
    navigate(buildRoutePath.historyDetail(id));
  };

  // Hardcode detection types for simplicity and stability
  const detectionTypes = ["all", "crack", "flatness"];

  if (loading) {
    return (
      <div className="lab-page flex justify-center items-center">
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      </div>
    );
  }

  return (
    <div className="lab-page">
      <div className="lab-container">
      <PageHeader
        title="检测历史"
        description="查看过往玻璃检测任务，按检测类型筛选并进入详情复核原始图、结果图和指标。"
        icon={<HistoryIcon className="w-8 h-8 text-white" />}
        iconBgClassName="bg-indigo-500"
      />
      <Tabs defaultValue="all" className="w-full">
        <TabsList>
          {detectionTypes.map(type => (
            <TabsTrigger key={type} value={type}>
              {type === 'all' ? '全部' : type === 'crack' ? '裂纹检测' : '平整度检测'}
            </TabsTrigger>
          ))}
        </TabsList>
        {detectionTypes.map(type => (
          <TabsContent key={type} value={type}>
            <div className="history-grid">
              {historyItems
                .filter(item => type === 'all' || item.type === type)
                .map((item) => (
                  <HistoryItemCard key={item.id} item={item} onSelect={() => handleItemClick(String(item.id))} />
                ))}
            </div>
            {historyItems.filter(item => type === 'all' || item.type === type).length === 0 && (
              <div className="history-empty">
                <p>没有找到相关记录。</p>
              </div>
            )}
          </TabsContent>
        ))}
      </Tabs>
      </div>
    </div>
  );
}
