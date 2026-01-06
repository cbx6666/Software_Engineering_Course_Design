import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { login as loginApi, register as registerApi, type LoginResponse } from "@/services/authApi";

const STORAGE_KEY = "glassdetect.auth";

export interface AuthState {
    user: LoginResponse["user"];
}

interface AuthContextType {
    auth: AuthState | null;
    isAuthed: boolean;
    isLoggingIn: boolean;
    login: (params: { email: string; password: string; remember: boolean }) => Promise<AuthState>;
    register: (params: { email: string; password: string; remember: boolean }) => Promise<AuthState>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

function readStoredAuth(): AuthState | null {
    const raw = localStorage.getItem(STORAGE_KEY) ?? sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
        const parsed = JSON.parse(raw) as any;
        if (parsed && typeof parsed === "object") {
            const user = parsed.user;
            if (user && typeof user === "object") {
                return { user } as AuthState;
            }
        }
        return null;
    } catch {
        return null;
    }
}

function writeStoredAuth(auth: AuthState, remember: boolean) {
    const raw = JSON.stringify(auth);
    if (remember) {
        localStorage.setItem(STORAGE_KEY, raw);
        sessionStorage.removeItem(STORAGE_KEY);
    } else {
        sessionStorage.setItem(STORAGE_KEY, raw);
        localStorage.removeItem(STORAGE_KEY);
    }
}

function clearStoredAuth() {
    localStorage.removeItem(STORAGE_KEY);
    sessionStorage.removeItem(STORAGE_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
    const [auth, setAuth] = useState<AuthState | null>(() => readStoredAuth());
    const [isLoggingIn, setIsLoggingIn] = useState(false);

    useEffect(() => {
        const onStorage = () => setAuth(readStoredAuth());
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, []);

    const isAuthed = !!auth?.user;

    const login = useCallback(async (params: { email: string; password: string; remember: boolean }) => {
        setIsLoggingIn(true);
        try {
            const res = await loginApi({ email: params.email, password: params.password });
            const next: AuthState = { user: res.user };
            writeStoredAuth(next, params.remember);
            setAuth(next);
            return next;
        } finally {
            setIsLoggingIn(false);
        }
    }, []);

    const register = useCallback(
        async (params: { email: string; password: string; remember: boolean }) => {
            setIsLoggingIn(true);
            try {
                const res = await registerApi({
                    email: params.email,
                    password: params.password,
                });
                const next: AuthState = { user: res.user };
                writeStoredAuth(next, params.remember);
                setAuth(next);
                return next;
            } finally {
                setIsLoggingIn(false);
            }
        },
        [],
    );

    const logout = useCallback(() => {
        clearStoredAuth();
        setAuth(null);
    }, []);

    const value = useMemo(
        () => ({
            auth,
            isAuthed,
            isLoggingIn,
            login,
            register,
            logout,
        }),
        [auth, isAuthed, isLoggingIn, login, register, logout],
    );

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
