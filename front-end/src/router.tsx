import { Navigate, createBrowserRouter, useNavigate } from "react-router-dom";
import App from "./App";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { GlassCrackDetectionPage } from "./pages/GlassCrackDetectionPage";
import { GlassFlatnessDetectionPage } from "./pages/GlassFlatnessDetectionPage";
import { HistoryPage } from "./pages/HistoryPage";
import { HistoryDetailPage } from "./pages/HistoryDetailPage";
import { useAuth } from "./hooks/useAuth";

function AuthPage() {
  const { login, register, isDevAuthBypass, isLoggingIn } = useAuth();
  const navigate = useNavigate();
  if (isDevAuthBypass) {
    return <Navigate to="/" replace />;
  }
  return (
    <LoginPage
      onLogin={async (params) => {
        await login({ ...params, remember: false });
        navigate("/");
      }}
      onRegister={async (params) => {
        await register({ ...params, remember: false });
        navigate("/");
      }}
      isSubmitting={isLoggingIn}
    />
  );
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true,
        element: <HomePage />,
      },
      {
        path: "crack",
        element: <GlassCrackDetectionPage />,
      },
      {
        path: "flatness",
        element: <GlassFlatnessDetectionPage />,
      },
      {
        path: "history",
        element: <HistoryPage />,
      },
      {
        path: "history/:id",
        element: <HistoryDetailPage />,
      },
    ],
  },
  {
    path: "/login",
    element: <AuthPage />,
  },
]);
