import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { GlassCrackDetectionPage } from "./pages/GlassCrackDetectionPage";
import { GlassFlatnessDetectionPage } from "./pages/GlassFlatnessDetectionPage";
import { HistoryPage } from "./pages/HistoryPage";

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
    ],
  },
  {
    path: "/login",
    element: <LoginPage />,
  },
]);

