import { createApiClient, postJson } from "./http";

export interface LoginRequest {
    email: string;
    password: string;
}

export interface RegisterRequest {
    email: string;
    password: string;
}

export interface LoginResponse {
    user: {
        id: string;
        email: string;
    };
}

/**
 * 登录接口（真实后端）
 * - POST /api/auth/login
 * - 返回：{ user: { id, email } }
 */
export async function login(req: LoginRequest): Promise<LoginResponse> {
    return await postJson<LoginResponse>(api, "/api/auth/login", req);
}

/**
 * 注册接口（真实后端）
 * - POST /api/auth/register
 * - 返回：{ user: { id, email } }
 */
export async function register(req: RegisterRequest): Promise<LoginResponse> {
    return await postJson<LoginResponse>(api, "/api/auth/register", req);
}

const api = createApiClient({
    // 你后端如果用 cookie/session，把这里改成 true 即可
    withCredentials: false,
});
