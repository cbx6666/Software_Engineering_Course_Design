export function ResultImage({ title = "平整度可视化", src }: { title?: string; src: string }) {
  return (
    <div className="mb-4">
      <h4 className="text-white text-sm font-medium mb-3">{title}</h4>
      <div className="rounded-lg overflow-hidden border border-white/20 bg-slate-900/50 shadow-lg">
        <img src={src} alt={title} className="w-full h-auto object-contain" />
      </div>
    </div>
  );
}


