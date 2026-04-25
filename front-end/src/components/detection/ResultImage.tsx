export function ResultImage({title = "平整度可视化", src}: { title?: string; src: string }) {
    const backendUrl = import.meta.env.VITE_BACKEND_URL || "";

    const fullSrc = src.startsWith("http")
        ? src
        : `${backendUrl}${src}`;

    return (
        <div className="result-media">
            <h4>{title}</h4>
            <div className="result-media__frame">
                <img
                    src={fullSrc}
                    alt={title}
                    onError={(e) => {
                        console.error("图片加载失败:", fullSrc);
                        e.currentTarget.style.display = 'none';
                    }}/>
            </div>
        </div>
    );
}
