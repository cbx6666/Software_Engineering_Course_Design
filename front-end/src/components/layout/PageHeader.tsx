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
    <div className="page-header">
      <div className={`page-header__icon ${iconBgClassName}`}>
        {icon}
      </div>
      <div className="page-header__text">
        <div className="eyebrow">INSPECTION TASK</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
    </div>
  );
}


