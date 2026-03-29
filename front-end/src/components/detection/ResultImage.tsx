export function ResultImage({title = "平整度可视化", src}: { title?: string; src: string }) {
    const backendUrl = import.meta.env.VITE_BACKEND_URL || "";

    const fullSrc = src.startsWith("http")
        ? src
        : `${backendUrl}${src}`;

    return (
        <div className="mb-4">
            <h4 className="text-white text-sm font-medium mb-3">{title}</h4>
            <div className="rounded-lg overflow-hidden border border-white/20 bg-slate-900/50 shadow-lg">
                <img
                    src={fullSrc}
                    alt={title}
                    className="w-full h-auto object-contain"
                    onError={(e) => {
                        console.error("图片加载失败:", fullSrc);
                        e.currentTarget.style.display = 'none';
                    }}/>
            </div>
        </div>
    );
}
