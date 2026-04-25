// 交互提示 + 重置按钮
export function ControlsPanel({ onReset }: { onReset: () => void }) {
  return (
    <div className="pointcloud-toolbar" style={{ pointerEvents: "auto" }}>
      <button
        onClick={onReset}
        type="button"
        className="pointcloud-reset"
      >
        重置视角
      </button>
    </div>
  );
}


