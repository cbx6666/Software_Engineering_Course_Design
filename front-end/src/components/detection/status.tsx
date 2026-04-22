import { AlertCircle, CheckCircle, Info } from "lucide-react";
import type { DetectionStatus } from "@/types/detection";

export function getStatusConfig(status: DetectionStatus) {
  switch (status) {
    case "success":
      return {
        icon: CheckCircle,
        badgeVariant: "default" as const,
        bgColor: "result-card--success",
        textColor: "text-green-400",
        borderColor: "",
        label: "正常",
      };
    case "warning":
      return {
        icon: Info,
        badgeVariant: "secondary" as const,
        bgColor: "result-card--warning",
        textColor: "text-yellow-400",
        borderColor: "",
        label: "警告",
      };
    case "error":
      return {
        icon: AlertCircle,
        badgeVariant: "destructive" as const,
        bgColor: "result-card--error",
        textColor: "text-red-400",
        borderColor: "",
        label: "异常",
      };
  }
}


