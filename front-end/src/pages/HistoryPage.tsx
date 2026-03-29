import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { HistoryItemCard } from "@/components/history/HistoryItemCard";
import { getHistory, type HistoryItem } from "@/services/historyApi";
import { useAuth } from "@/hooks/useAuth";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/CustomTabs";
import { History as HistoryIcon, Loader2 } from "lucide-react";

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
    navigate(`/history/${id}`);
  };

  // Hardcode detection types for simplicity and stability
  const detectionTypes = ["all", "crack", "flatness"];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 md:p-6 lg:p-8">
      <PageHeader
        title="检测历史"
        description="查看过去的所有检测记录"
        icon={<HistoryIcon className="w-8 h-8 text-white" />}
        iconBgClassName="bg-indigo-500"
      />
      <Tabs defaultValue="all" className="w-full mt-6">
        <TabsList>
          {detectionTypes.map(type => (
            <TabsTrigger key={type} value={type}>
              {type === 'all' ? '全部' : type === 'crack' ? '裂纹检测' : '平整度检测'}
            </TabsTrigger>
          ))}
        </TabsList>
        {detectionTypes.map(type => (
          <TabsContent key={type} value={type}>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 mt-4">
              {historyItems
                .filter(item => type === 'all' || item.type === type)
                .map((item) => (
                  <HistoryItemCard key={item.id} item={item} onSelect={() => handleItemClick(String(item.id))} />
                ))}
            </div>
            {historyItems.filter(item => type === 'all' || item.type === type).length === 0 && (
              <div className="text-center text-slate-400 py-10">
                <p>没有找到相关记录。</p>
              </div>
            )}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
