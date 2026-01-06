import type { HistoryItem } from "@/services/historyApi";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScanSearch, Ruler, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

interface HistoryItemCardProps {
  item: HistoryItem;
}

const typeDetails: Record<HistoryItem['type'], { name: string; icon: React.ReactNode }> = {
  crack: { name: "玻璃破裂检测", icon: <ScanSearch className="w-4 h-4" /> },
  flatness: { name: "幕墙平整度检测", icon: <Ruler className="w-4 h-4" /> },
};

const statusDetails: Record<HistoryItem['status'], { variant: 'success' | 'warning' | 'destructive'; icon: React.ReactNode }> = {
  success: { variant: "success", icon: <CheckCircle2 className="w-4 h-4 mr-2" /> },
  warning: { variant: "warning", icon: <AlertTriangle className="w-4 h-4 mr-2" /> },
  error: { variant: "destructive", icon: <XCircle className="w-4 h-4 mr-2" /> },
};

export function HistoryItemCard({ item, onSelect }: HistoryItemCardProps & { onSelect?: (item: HistoryItem) => void }) {
  const { name, icon } = typeDetails[item.type];
  const { variant, icon: statusIcon } = statusDetails[item.status];

  return (
    <Card 
      onClick={() => onSelect?.(item)}
      className="border-0 bg-white/10 backdrop-blur-xl shadow-lg overflow-hidden transition-transform duration-300 hover:-translate-y-1 hover:shadow-cyan-500/20 cursor-pointer"
    >
      <div className="flex flex-col">
        {/* Content */}
        <div className="flex-1 p-6">
          {/* Header */}
          <div className="flex justify-between items-start mb-3">
            <div className="flex items-center text-cyan-400 text-sm">
              {icon}
              <span className="ml-2 font-semibold">{name}</span>
            </div>
            <Badge variant={variant} className="capitalize">
              {statusIcon}
              {item.status}
            </Badge>
          </div>

          {/* Title & Date */}
          <h3 className="text-white font-bold text-lg mb-1">{item.title}</h3>
          <p className="text-slate-400 text-xs mb-4">
            {new Date(item.date).toLocaleString('zh-CN')}
          </p>

          {/* Description */}
          <p className="text-slate-300 text-sm leading-relaxed">
            {item.description}
          </p>
        </div>
      </div>
    </Card>
  );
}

