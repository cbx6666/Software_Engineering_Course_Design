export const ROUTE_PATHS = {
  HOME: "/",
  LOGIN: "/login",
  CRACK: "/crack",
  FLATNESS: "/flatness",
  HISTORY: "/history",
  HISTORY_DETAIL: "/history/:id",
} as const;

export const ROUTE_SEGMENTS = {
  CRACK: "crack",
  FLATNESS: "flatness",
  HISTORY: "history",
  HISTORY_DETAIL: "history/:id",
} as const;

export const buildRoutePath = {
  historyDetail: (id: string | number) => `${ROUTE_PATHS.HISTORY}/${id}`,
};

export function isHomeRoute(pathname: string) {
  return pathname === ROUTE_PATHS.HOME;
}
