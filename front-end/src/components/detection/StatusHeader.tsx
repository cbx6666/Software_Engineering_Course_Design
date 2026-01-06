import { Badge } from "@/components/ui/badge";
import type { DetectionResultData } from "@/types/detection";
import { getStatusConfig } from "@/components/detection/status";

export function StatusHeader({ result }: { result: Pick<DetectionResultData, "status" | "title" | "description"> }) {
  const config = getStatusConfig(result.status);
  const Icon = config.icon;

  return (
    <div className="flex items-start gap-4">
      <div className={config.textColor}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-3 mb-2">
          <h3 className="text-white">{result.title}</h3>
          <Badge variant={config.badgeVariant}>{config.label}</Badge>
        </div>
        <p className="text-slate-300 mb-4">{result.description}</p>
      </div>
    </div>
  );
}


