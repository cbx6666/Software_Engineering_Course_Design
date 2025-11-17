import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { AlertCircle, CheckCircle, Info } from "lucide-react";

// 检测结果定义在此处
export interface DetectionResultData {
  status: "success" | "warning" | "error";
  title: string;
  description: string;
  details?: Array<{ label: string; value: string }>;
}

interface DetectionResultProps {
  result: DetectionResultData;
}

export function DetectionResult({ result }: DetectionResultProps) {
  const getStatusConfig = () => {
    switch (result.status) {
      case "success":
        return {
          icon: <CheckCircle className="w-5 h-5" />,
          badgeVariant: "default" as const,
          bgColor: "bg-green-500/10 backdrop-blur-md",
          textColor: "text-green-400",
          borderColor: "border-green-500/30"
        };
      case "warning":
        return {
          icon: <Info className="w-5 h-5" />,
          badgeVariant: "secondary" as const,
          bgColor: "bg-yellow-500/10 backdrop-blur-md",
          textColor: "text-yellow-400",
          borderColor: "border-yellow-500/30"
        };
      case "error":
        return {
          icon: <AlertCircle className="w-5 h-5" />,
          badgeVariant: "destructive" as const,
          bgColor: "bg-red-500/10 backdrop-blur-md",
          textColor: "text-red-400",
          borderColor: "border-red-500/30"
        };
    }
  };

  const config = getStatusConfig();

  return (
    <Card className={`p-6 border ${config.borderColor} ${config.bgColor}`}>
      <div className="flex items-start gap-4">
        <div className={config.textColor}>{config.icon}</div>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className={`text-white`}>{result.title}</h3>
            <Badge variant={config.badgeVariant}>
              {result.status === "success" && "正常"}
              {result.status === "warning" && "警告"}
              {result.status === "error" && "异常"}
            </Badge>
          </div>
          <p className="text-slate-300 mb-4">{result.description}</p>
          
          {result.details && result.details.length > 0 && (
            <div className="space-y-2 bg-slate-900/30 rounded-lg p-4 backdrop-blur-sm">
              {result.details.map((detail, index) => (
                <div key={index} className="flex justify-between items-center py-2 border-t border-white/10 first:border-t-0">
                  <span className="text-slate-400">{detail.label}</span>
                  <span className="text-white">{detail.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}