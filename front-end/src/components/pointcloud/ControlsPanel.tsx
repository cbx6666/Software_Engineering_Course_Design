// 交互提示 + 重置按钮
export function ControlsPanel({ onReset }: { onReset: () => void }) {
  return (
    <div className="absolute top-2 left-6 z-50 flex flex-col gap-2 pointer-events-auto" style={{ pointerEvents: "auto" }}>
      <button
        onClick={onReset}
        type="button"
        className="px-4 py-2 text-xs font-medium rounded-lg hover:bg-white/10 transition-colors"
        style={{
          background: "transparent",
          border: "1px solid rgba(255, 255, 255, 0.6)",
          color: "#ffffff",
          fontFamily: "Microsoft YaHei",
          fontWeight: 400,
          fontSize: 15,
        }}
      >
        重置视角
      </button>
    </div>
  );
}


