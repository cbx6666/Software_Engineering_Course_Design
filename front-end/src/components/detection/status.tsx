import { AlertCircle, CheckCircle, Info } from "lucide-react";
import type { DetectionStatus } from "@/types/detection";

export function getStatusConfig(status: DetectionStatus) {
  switch (status) {
    case "success":
      return {
        icon: CheckCircle,
        badgeVariant: "default" as const,
        bgColor: "bg-green-500/10 backdrop-blur-md",
        textColor: "text-green-400",
        borderColor: "border-green-500/30",
        label: "正常",
      };
    case "warning":
      return {
        icon: Info,
        badgeVariant: "secondary" as const,
        bgColor: "bg-yellow-500/10 backdrop-blur-md",
        textColor: "text-yellow-400",
        borderColor: "border-yellow-500/30",
        label: "警告",
      };
    case "error":
      return {
        icon: AlertCircle,
        badgeVariant: "destructive" as const,
        bgColor: "bg-red-500/10 backdrop-blur-md",
        textColor: "text-red-400",
        borderColor: "border-red-500/30",
        label: "异常",
      };
  }
}


