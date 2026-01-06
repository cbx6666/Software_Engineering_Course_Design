import { useEffect, useState } from "react";
import { HomePage } from "@/components/HomePage";
import { GlassCrackDetection } from "@/components/GlassCrackDetection";
import { GlassFlatnessDetection } from "@/components/GlassFlatnessDetection";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LoginPage } from "@/components/auth/LoginPage";
import { useAuth } from "@/hooks/useAuth";
import { LogOut } from "lucide-react";

type Page = "login" | "home" | "crack" | "flatness";

export default function App() {
  const { isAuthed, login, register, logout, isLoggingIn } = useAuth();
  const [currentPage, setCurrentPage] = useState<Page>("login");

  useEffect(() => {
    if (!isAuthed) {
      setCurrentPage("login");
      return;
    }
    setCurrentPage((p) => (p === "login" ? "home" : p));
  }, [isAuthed]);

  const renderPage = () => {
    if (!isAuthed && currentPage !== "login") return null;

    switch (currentPage) {
      case "login":
        return (
          <LoginPage
            isSubmitting={isLoggingIn}
            onLogin={async (params) => {
              await login({ ...params, remember: true });
              setCurrentPage("home");
            }}
            onRegister={async (params) => {
              await register({ ...params, remember: true });
              setCurrentPage("home");
            }}
          />
        );
      case "home":
        return <HomePage onNavigate={(page) => setCurrentPage(page)} />;
      case "crack":
        return <GlassCrackDetection />;
      case "flatness":
        return <GlassFlatnessDetection />;
      default:
        return <HomePage onNavigate={(page) => setCurrentPage(page)} />;
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
      {isAuthed && currentPage !== "home" && currentPage !== "login" && (
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

      {/* Logout（仅主页显示，样式与返回按钮一致，位置在左上角偏右） */}
      {isAuthed && currentPage === "home" && (
        <div className="absolute top-6 left-6 z-50">
          <Button
            onClick={() => {
              logout();
              setCurrentPage("login");
            }}
            variant="outline"
            className="bg-white/10 backdrop-blur-md border-white/20 text-white hover:bg-white/20"
          >
            <LogOut className="w-4 h-4 mr-2" />
            退出登录
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
