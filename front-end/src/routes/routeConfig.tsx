import type { RouteObject } from "react-router-dom";
import { Navigate } from "react-router-dom";
import App from "@/App";
import { HomePage } from "@/pages/HomePage";
import { GlassCrackDetectionPage } from "@/pages/GlassCrackDetectionPage";
import { GlassFlatnessDetectionPage } from "@/pages/GlassFlatnessDetectionPage";
import { HistoryPage } from "@/pages/HistoryPage";
import { HistoryDetailPage } from "@/pages/HistoryDetailPage";
import { AuthPage } from "@/routes/AuthPage";
import { ROUTE_PATHS, ROUTE_SEGMENTS } from "@/routes/paths";

export interface RouteMeta {
  title: string;
  group: "dashboard" | "detection" | "records" | "auth";
  showInNavigation?: boolean;
}

export type ManagedRouteObject = RouteObject & {
  meta: RouteMeta;
};

export const protectedPageRoutes: ManagedRouteObject[] = [
  {
    index: true,
    element: <HomePage />,
    meta: {
      title: "工作台首页",
      group: "dashboard",
      showInNavigation: true,
    },
  },
  {
    path: ROUTE_SEGMENTS.CRACK,
    element: <GlassCrackDetectionPage />,
    meta: {
      title: "玻璃破裂检测",
      group: "detection",
      showInNavigation: true,
    },
  },
  {
    path: ROUTE_SEGMENTS.FLATNESS,
    element: <GlassFlatnessDetectionPage />,
    meta: {
      title: "幕墙平整度检测",
      group: "detection",
      showInNavigation: true,
    },
  },
  {
    path: ROUTE_SEGMENTS.HISTORY,
    element: <HistoryPage />,
    meta: {
      title: "检测历史",
      group: "records",
      showInNavigation: true,
    },
  },
  {
    path: ROUTE_SEGMENTS.HISTORY_DETAIL,
    element: <HistoryDetailPage />,
    meta: {
      title: "检测详情",
      group: "records",
    },
  },
];

export const publicPageRoutes: ManagedRouteObject[] = [
  {
    path: ROUTE_PATHS.LOGIN,
    element: <AuthPage />,
    meta: {
      title: "登录",
      group: "auth",
    },
  },
];

export const appRoutes: RouteObject[] = [
  {
    path: ROUTE_PATHS.HOME,
    element: <App />,
    children: protectedPageRoutes,
  },
  ...publicPageRoutes,
  {
    path: "*",
    element: <Navigate to={ROUTE_PATHS.HOME} replace />,
  },
];
