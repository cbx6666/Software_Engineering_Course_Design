import { useState } from "react";
import { HomePage } from "./components/HomePage";
import { GlassCrackDetection } from "./components/GlassCrackDetection";
import { GlassFlatnessDetection } from "./components/GlassFlatnessDetection";
import { ArrowLeft } from "lucide-react";
import { Button } from "./components/ui/button";

type Page = "home" | "crack" | "flatness";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={setCurrentPage} />;
      case "crack":
        return <GlassCrackDetection />;
      case "flatness":
        return <GlassFlatnessDetection />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative overflow-hidden">
      {/* Glass Pattern Background */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px'
        }}></div>
      </div>

      {/* Animated Glass Shards */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Back Button */}
      {currentPage !== "home" && (
        <div className="absolute top-6 left-6 z-50">
          <Button
            onClick={() => setCurrentPage("home")}
            variant="outline"
            className="bg-white/10 backdrop-blur-md border-white/20 text-white hover:bg-white/20"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回主页
          </Button>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10">
        {renderPage()}
      </div>
    </div>
  );
}
