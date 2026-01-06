import { useCallback, useEffect, useMemo, useState } from "react";
import { login as loginApi, register as registerApi, type LoginResponse } from "@/services/authApi";

const STORAGE_KEY = "glassdetect.auth";

export interface AuthState {
    user: LoginResponse["user"];
}

function readStoredAuth(): AuthState | null {
    const raw = localStorage.getItem(STORAGE_KEY) ?? sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
        const parsed = JSON.parse(raw) as any;
        // 兼容旧版本：{ token, user }
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

export function useAuth() {
    const [auth, setAuth] = useState<AuthState | null>(() => readStoredAuth());
    const [isLoggingIn, setIsLoggingIn] = useState(false);

    // 同步外部（比如手动清 storage）情况
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

    return useMemo(
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
}


