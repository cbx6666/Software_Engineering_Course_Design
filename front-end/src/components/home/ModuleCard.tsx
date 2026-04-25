import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { ChevronRight } from "lucide-react";

export interface ModuleCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  onClick?: () => void;
  gradient: string;
  iconBg: string;
  comingSoon?: boolean;
}

export function ModuleCard({ icon, title, description, onClick, gradient, iconBg, comingSoon }: ModuleCardProps) {
  return (
    <Card
      className={`module-card ${comingSoon ? "module-card--muted" : ""}`}
      onClick={comingSoon ? undefined : onClick}
      data-gradient={gradient}
      data-icon-bg={iconBg}
    >
      <div className="module-card__inner">
        <div>
          <div className="module-card__top">
            <div className="module-card__icon">{icon}</div>
            <span className="module-card__code">QC-{title.length.toString().padStart(2, "0")}</span>
          </div>

          <div>
            <h3>{title}</h3>
            <p>{description}</p>
          </div>
        </div>

        <div className="module-card__footer">
          <span>{comingSoon ? "等待接入" : "进入检测流程"}</span>
          {!comingSoon && <ChevronRight />}
        </div>

        {!comingSoon && (
          <div className="module-card__trace" />
        )}
      </div>
    </Card>
  );
}
