import { useEffect } from "react";
import { ArrowLeft, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/useAuth";
import { ROUTE_PATHS, isHomeRoute } from "@/routes";
import { Outlet, useLocation, useNavigate } from "react-router-dom";

export default function App() {
  const { isAuthed, isDevAuthBypass, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!isAuthed && !isDevAuthBypass) {
      navigate(ROUTE_PATHS.LOGIN);
    }
  }, [isAuthed, isDevAuthBypass, navigate]);

  const isHomePage = isHomeRoute(location.pathname);

  return (
    <div className="glass-app">
      <div className="shell-scanline" />

      {!isHomePage && (
        <div className="shell-action">
          <Button
            onClick={() => navigate(ROUTE_PATHS.HOME)}
            variant="outline"
            className="shell-button"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回主页
          </Button>
        </div>
      )}

      {isAuthed && isHomePage && !isDevAuthBypass && (
        <div className="shell-action">
          <Button
            onClick={() => {
              logout();
            }}
            variant="outline"
            className="shell-button"
          >
            <LogOut className="w-4 h-4 mr-2" />
            退出登录
          </Button>
        </div>
      )}

      <div className="relative z-10 min-h-screen">
        <Outlet />
      </div>
    </div>
  );
}
