import { Building2, CircuitBoard, Layers3, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { ModuleCard } from "@/components/home/ModuleCard";
import { getHomeModules } from "@/components/home/modules";
import { GlassInspectionVisual } from "@/components/visual/GlassInspectionVisual";

export function HomePage() {
  const navigate = useNavigate();
  const modules = getHomeModules(navigate);

  const metrics = [
    { value: "3D", label: "幕墙平整度复核" },
    { value: "1px", label: "裂纹边缘追踪" },
    { value: "24h", label: "检测记录追溯" },
  ];

  return (
    <div className="lab-page lab-page--home">
      <div className="lab-container">
        <section className="home-hero">
          <div className="hero-copy-block">
            <div className="eyebrow">
              <Building2 />
              玻璃幕墙质检工作台
            </div>

            <h1 className="hero-title">
              玻璃智能检测
              <span>可视化系统</span>
            </h1>
            <p className="hero-copy">
              面向玻璃裂纹、破损与幕墙平整度检测的现场工作台。上传图像后即可完成缺陷判断、结果复核、3D 点云查看与历史归档。
            </p>

            <div className="tag-row">
              {[
                { icon: <ShieldCheck className="w-4 h-4" />, text: "裂纹缺陷识别" },
                { icon: <Layers3 className="w-4 h-4" />, text: "3D 点云地图" },
                { icon: <CircuitBoard className="w-4 h-4" />, text: "批次结果追踪" },
              ].map((tag) => (
                <span className="tag-pill" key={tag.text}>
                  {tag.icon}
                  {tag.text}
                </span>
              ))}
            </div>

            <div className="hero-metrics">
              {metrics.map((metric) => (
                <div className="metric-tile" key={metric.label}>
                  <strong>{metric.value}</strong>
                  <span>{metric.label}</span>
                </div>
              ))}
            </div>
          </div>

          <GlassInspectionVisual />
        </section>

        <div className="section-head">
          <div>
            <h2>检测模块</h2>
            <p>按任务进入对应流程，所有结果都会写入历史记录。</p>
          </div>
        </div>

        <div className="module-grid">
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
