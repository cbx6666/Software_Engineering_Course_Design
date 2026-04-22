import type { ReactNode } from 'react';
import type { HistoryItem } from "@/services/historyApi";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScanSearch, Ruler, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

interface HistoryItemCardProps {
  item: HistoryItem;
}

const typeDetails: Record<HistoryItem['type'], { name: string; icon: ReactNode }> = {
  crack: { name: "玻璃破裂检测", icon: <ScanSearch className="w-4 h-4" /> },
  flatness: { name: "幕墙平整度检测", icon: <Ruler className="w-4 h-4" /> },
};

const statusDetails: Record<HistoryItem['status'], { variant: 'success' | 'warning' | 'destructive'; icon: ReactNode; label: string }> = {
  success: { variant: "success", icon: <CheckCircle2 className="w-4 h-4 mr-2" />, label: "正常" },
  warning: { variant: "warning", icon: <AlertTriangle className="w-4 h-4 mr-2" />, label: "警告" },
  error: { variant: "destructive", icon: <XCircle className="w-4 h-4 mr-2" />, label: "异常" },
};

export function HistoryItemCard({ item, onSelect }: HistoryItemCardProps & { onSelect?: (item: HistoryItem) => void }) {
  const { name, icon } = typeDetails[item.type];
  const { variant, icon: statusIcon, label: statusLabel } = statusDetails[item.status];

  return (
    <Card 
      onClick={() => onSelect?.(item)}
      className="history-card"
    >
      <div className="history-card__body">
          <div className="history-card__meta">
            <div className="history-card__type">
              {icon}
              <span>{name}</span>
            </div>
            <Badge variant={variant} className="status-badge">
              {statusIcon}
              {statusLabel}
            </Badge>
          </div>

          <h3>{item.title}</h3>
          <time>
            {new Date(item.date).toLocaleString('zh-CN')}
          </time>

          <p>{item.description}</p>
      </div>
    </Card>
  );
}
