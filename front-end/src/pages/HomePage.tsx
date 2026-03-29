import { Building2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { ModuleCard } from "@/components/home/ModuleCard";
import { getHomeModules } from "@/components/home/modules";

export function HomePage() {
  const navigate = useNavigate();
  const modules = getHomeModules(navigate);

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
          {modules.map((module, idx) => (
            <ModuleCard
              key={idx}
              icon={module.icon}
              title={module.title}
              description={module.description}
              gradient={module.gradient}
              iconBg={module.iconBg}
              onClick={module.onClick}
              comingSoon={module.comingSoon}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
