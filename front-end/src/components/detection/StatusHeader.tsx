import { Badge } from "@/components/ui/badge";
import type { DetectionResultData } from "@/types/detection";
import { getStatusConfig } from "@/components/detection/status";

export function StatusHeader({ result }: { result: Pick<DetectionResultData, "status" | "title" | "description"> }) {
  const config = getStatusConfig(result.status);
  const Icon = config.icon;

  return (
    <div className="status-header">
      <div className={`status-icon ${config.textColor}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <div className="status-title-row">
          <h3>{result.title}</h3>
          <Badge variant={config.badgeVariant} className="status-badge">{config.label}</Badge>
        </div>
        <p>{result.description}</p>
      </div>
    </div>
  );
}


