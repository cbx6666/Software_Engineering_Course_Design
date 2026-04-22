import { ScanSearch, Ruler, History } from "lucide-react";
import type { ModuleCardProps } from "@/components/home/ModuleCard";
import type { NavigateFunction } from "react-router-dom";
import { ROUTE_PATHS } from "@/routes";

export function getHomeModules(navigate: NavigateFunction): ModuleCardProps[] {
  return [
    {
      icon: <ScanSearch className="w-8 h-8" />,
      title: "玻璃破裂检测",
      description: "上传玻璃表面图像，定位裂纹、崩边与破损区域，输出可复核的检测结论。",
      gradient: "from-cyan-500/40 via-blue-500/40 to-indigo-500/40",
      iconBg: "from-cyan-500 to-blue-600",
      onClick: () => navigate(ROUTE_PATHS.CRACK),
    },
    {
      icon: <Ruler className="w-8 h-8" />,
      title: "幕墙平整度检测",
      description: "按左右环境图与投影图生成平整度判断，并保留 3D 点云地图辅助复查。",
      gradient: "from-blue-500/40 via-indigo-500/40 to-purple-500/40",
      iconBg: "from-blue-500 to-indigo-600",
      onClick: () => navigate(ROUTE_PATHS.FLATNESS),
    },
    {
      icon: <History className="w-8 h-8" />,
      title: "历史检测记录",
      description: "追溯过往检测任务、原始图片、结果图与指标明细，便于批次管理。",
      gradient: "from-purple-500/40 via-pink-500/40 to-rose-500/40",
      iconBg: "from-purple-500 to-pink-600",
      onClick: () => navigate(ROUTE_PATHS.HISTORY),
    },
  ];
}
