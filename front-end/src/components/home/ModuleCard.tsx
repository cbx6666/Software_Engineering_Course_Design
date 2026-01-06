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
      className={`group relative overflow-hidden border-0 bg-gradient-to-br ${gradient} p-1 transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-cyan-500/20 ${comingSoon ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
      onClick={comingSoon ? undefined : onClick}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative bg-slate-900/90 backdrop-blur-xl rounded-lg p-8 h-full">
        <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${iconBg} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
          <div className="text-white">{icon}</div>
        </div>

        <div className="space-y-3">
          <h3 className="text-white flex items-center justify-between">
            {title}
            {!comingSoon && <ChevronRight className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform duration-300" />}
          </h3>
          <p className="text-slate-300 leading-relaxed">{description}</p>
        </div>

        {!comingSoon && (
          <div
          className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-blue-500
                     transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-center"
        />
        )}
      </div>
    </Card>
  );
}
