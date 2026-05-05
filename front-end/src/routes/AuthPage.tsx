import { Navigate, useNavigate } from "react-router-dom";
import { LoginPage } from "@/pages/LoginPage";
import { useAuth } from "@/hooks/useAuth";
import { ROUTE_PATHS } from "@/routes/paths";

export function AuthPage() {
  const { login, register, isDevAuthBypass, isLoggingIn } = useAuth();
  const navigate = useNavigate();

  // if (isDevAuthBypass) {
  //   return <Navigate to={ROUTE_PATHS.HOME} replace />;
  // }

  return (
    <LoginPage
      onLogin={async (params) => {
        await login({ ...params, remember: false });
        navigate(ROUTE_PATHS.HOME);
      }}
      onRegister={async (params) => {
        await register({ ...params, remember: false });
        navigate(ROUTE_PATHS.HOME);
      }}
      isSubmitting={isLoggingIn}
    />
  );
}
