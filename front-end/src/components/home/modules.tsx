import { ScanSearch, Ruler, Sparkles } from "lucide-react";
import type { ModuleCardProps } from "@/components/home/ModuleCard";

export type HomePageRoute = "crack" | "flatness";

export function getHomeModules(onNavigate: (page: HomePageRoute) => void): ModuleCardProps[] {
  return [
    {
      icon: <ScanSearch className="w-8 h-8" />,
      title: "玻璃破裂检测",
      description: "基于AI视觉技术，精准识别玻璃表面的裂纹、破损等缺陷，支持多种破裂模式检测",
      gradient: "from-cyan-500/40 via-blue-500/40 to-indigo-500/40",
      iconBg: "from-cyan-500 to-blue-600",
      onClick: () => onNavigate("crack"),
    },
    {
      icon: <Ruler className="w-8 h-8" />,
      title: "幕墙平整度检测",
      description: "智能分析玻璃幕墙表面平整度，快速定位凹凸异常区域，提供精确的偏差数据",
      gradient: "from-blue-500/40 via-indigo-500/40 to-purple-500/40",
      iconBg: "from-blue-500 to-indigo-600",
      onClick: () => onNavigate("flatness"),
    },
    {
      icon: <Sparkles className="w-8 h-8" />,
      title: "表面缺陷检测",
      description: "精细化检测玻璃表面划痕、污渍、杂质等细微缺陷，确保产品完美品质",
      gradient: "from-purple-500/40 via-pink-500/40 to-rose-500/40",
      iconBg: "from-purple-500 to-pink-600",
      comingSoon: true,
    },
  ];
}


