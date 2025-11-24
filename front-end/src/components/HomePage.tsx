import { Card } from "./ui/card";
import { 
  ScanSearch, 
  Ruler, 
  Sparkles, 
  Building2,
  ChevronRight
} from "lucide-react";

interface ModuleCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
  gradient: string;
  iconBg: string;
  comingSoon?: boolean;
}

function ModuleCard({ icon, title, description, onClick, gradient, iconBg, comingSoon }: ModuleCardProps) {
  return (
    <Card 
      className={`group relative overflow-hidden border-0 bg-gradient-to-br ${gradient} p-1 transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-cyan-500/20 ${comingSoon ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
      onClick={comingSoon ? undefined : onClick}
    >
      {/* Glass Effect Border */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
      
      {/* Inner Card */}
      <div className="relative bg-slate-900/90 backdrop-blur-xl rounded-lg p-8 h-full">
        {/* Icon */}
        <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${iconBg} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
          <div className="text-white">
            {icon}
          </div>
        </div>

        {/* Content */}
        <div className="space-y-3">
          <h3 className="text-white flex items-center justify-between">
            {title}
            {!comingSoon && (
              <ChevronRight className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform duration-300" />
            )}
          </h3>
          <p className="text-slate-300 leading-relaxed">
            {description}
          </p>
        </div>

        {/* Coming Soon Badge */}
        {comingSoon && (
          <div className="absolute top-4 right-4">
            <span className="px-3 py-1 bg-yellow-500/20 text-yellow-300 rounded-full text-sm backdrop-blur-sm border border-yellow-500/30">
              即将上线
            </span>
          </div>
        )}

        {/* Hover Indicator */}
        {!comingSoon && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
        )}
      </div>
    </Card>
  );
}

interface HomePageProps {
  onNavigate: (page: "crack" | "flatness") => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  const modules = [
    {
      id: "crack",
      icon: <ScanSearch className="w-8 h-8" />,
      title: "玻璃破裂检测",
      description: "基于AI视觉技术，精准识别玻璃表面的裂纹、破损等缺陷，支持多种破裂模式检测",
      gradient: "from-cyan-500/40 via-blue-500/40 to-indigo-500/40",
      iconBg: "from-cyan-500 to-blue-600",
      onClick: () => onNavigate("crack")
    },
    {
      id: "flatness",
      icon: <Ruler className="w-8 h-8" />,
      title: "幕墙平整度检测",
      description: "智能分析玻璃幕墙表面平整度，快速定位凹凸异常区域，提供精确的偏差数据",
      gradient: "from-blue-500/40 via-indigo-500/40 to-purple-500/40",
      iconBg: "from-blue-500 to-indigo-600",
      onClick: () => onNavigate("flatness")
    },
    {
      id: "defect",
      icon: <Sparkles className="w-8 h-8" />,
      title: "表面缺陷检测",
      description: "精细化检测玻璃表面划痕、污渍、杂质等细微缺陷，确保产品完美品质",
      gradient: "from-purple-500/40 via-pink-500/40 to-rose-500/40",
      iconBg: "from-purple-500 to-pink-600",
      comingSoon: true
    },

  ];

  return (
    <div className="min-h-screen p-8 md:p-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 relative">
          {/* Glass Effect Header Background */}
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-96 h-96 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full blur-3xl"></div>
          
          <div className="relative">
            {/* Logo/Icon */}
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-cyan-500 to-blue-600 mb-6 shadow-2xl shadow-cyan-500/30">
              <Building2 className="w-10 h-10 text-white" />
            </div>

            <h1 className="text-white mb-4 bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
              玻璃智能检测系统
            </h1>
            <p className="text-slate-300 max-w-2xl mx-auto leading-relaxed">
              基于深度学习的玻璃质量检测平台，提供全方位、高精度的玻璃及幕墙检测服务
            </p>

            {/* Feature Tags */}
            <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
              {["AI智能识别", "高精度检测", "实时分析", "云端处理"].map((tag, index) => (
                <span 
                  key={index}
                  className="px-4 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-full text-slate-200 text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module) => (
            <ModuleCard
              key={module.id}
              icon={module.icon}
              title={module.title}
              description={module.description}
              gradient={module.gradient}
              iconBg={module.iconBg}
              onClick={module.onClick || (() => {})}
              comingSoon={module.comingSoon}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
