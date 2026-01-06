import { PageHeader } from "@/components/layout/PageHeader";
import { History } from "lucide-react";
import { HistoryItemCard } from "@/components/history/HistoryItemCard";

// Mock data for demonstration
const mockHistoryItems = [
  {
    id: "1",
    date: "2024-05-20T10:30:00Z",
    type: "平整度检测",
    result: "合格",
    imageUrl: "/placeholder.svg", // Replace with actual image path or URL
  },
  {
    id: "2",
    date: "2024-05-19T15:00:00Z",
    type: "裂纹检测",
    result: "不合格",
    details: {
      "裂纹数量": 3,
      "最大裂纹长度": "1.2mm",
    },
    imageUrl: "/placeholder.svg", // Replace with actual image path or URL
  },
];

export function HistoryPage() {
  return (
    <div className="container mx-auto p-4 md:p-6 lg:p-8">
      <PageHeader
        title="检测历史"
        description="查看和管理过去的所有检测记录。"
        icon={<History className="w-8 h-8 text-white" />}
        iconBgClassName="bg-indigo-500"
      />
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {mockHistoryItems.map((item) => (
          <HistoryItemCard key={item.id} item={item as any} />
        ))}
      </div>
    </div>
  );
}

