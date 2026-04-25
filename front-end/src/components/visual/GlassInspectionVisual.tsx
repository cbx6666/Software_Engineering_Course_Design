import { Activity, Gauge, ScanLine, Sparkles } from "lucide-react";

export function GlassInspectionVisual() {
  const chips = [
    { label: "Crack", value: "0.18mm", tone: "warning" },
    { label: "Warp", value: "3D Map", tone: "success" },
    { label: "Batch", value: "Live", tone: "info" },
  ];

  return (
    <div className="inspection-visual" aria-hidden="true">
      <div className="inspection-visual__stage">
        <div className="inspection-visual__rail inspection-visual__rail--top" />
        <div className="inspection-visual__rail inspection-visual__rail--bottom" />
        <div className="inspection-visual__camera">
          <ScanLine className="inspection-visual__camera-icon" />
        </div>
        <div className="inspection-visual__sheet">
          <div className="inspection-visual__scan" />
          <div className="inspection-visual__crack inspection-visual__crack--one" />
          <div className="inspection-visual__crack inspection-visual__crack--two" />
          <div className="inspection-visual__heat inspection-visual__heat--one" />
          <div className="inspection-visual__heat inspection-visual__heat--two" />
          <div className="inspection-visual__corner inspection-visual__corner--tl" />
          <div className="inspection-visual__corner inspection-visual__corner--br" />
        </div>
        <div className="inspection-visual__panel">
          <div className="inspection-visual__panel-head">
            <Sparkles className="inspection-visual__panel-icon" />
            <span>VISION LINE</span>
          </div>
          <div className="inspection-visual__bars">
            <span />
            <span />
            <span />
          </div>
          <div className="inspection-visual__meter">
            <Gauge />
            <strong>98.7%</strong>
          </div>
        </div>
        <div className="inspection-visual__wave">
          <Activity />
        </div>
      </div>

      <div className="inspection-visual__chips">
        {chips.map((chip) => (
          <div className={`inspection-chip inspection-chip--${chip.tone}`} key={chip.label}>
            <span>{chip.label}</span>
            <strong>{chip.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}
