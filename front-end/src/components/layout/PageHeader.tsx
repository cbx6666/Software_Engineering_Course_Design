import type { ReactNode } from "react";

export function PageHeader({
  icon,
  title,
  description,
  iconBgClassName,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  iconBgClassName: string;
}) {
  return (
    <div className="text-center mb-12">
      <div className={`inline-flex items-center justify-center w-16 h-16 rounded-2xl ${iconBgClassName} mb-4 shadow-xl`}>
        {icon}
      </div>
      <h1 className="text-white mb-3">{title}</h1>
      <p className="text-slate-300 max-w-2xl mx-auto">{description}</p>
    </div>
  );
}


