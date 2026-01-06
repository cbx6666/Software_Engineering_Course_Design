import { createApiClient, postJson } from "./http";

export interface LoginRequest {
    username: string;
    password: string;
}

export interface RegisterRequest {
    username: string;
    password: string;
    displayName?: string;
}

export interface LoginResponse {
    token: string;
    user: {
        username: string;
        displayName?: string;
    };
}

/**
 * 登录接口（真实后端）
 * - POST /api/auth/login
 * - 返回：{ token, user: { username, displayName? } }
 */
export async function login(req: LoginRequest): Promise<LoginResponse> {
    return await postJson<LoginResponse>(api, "/api/auth/login", req);
}

/**
 * 注册接口（真实后端）
 * - POST /api/auth/register
 * - 返回：{ token, user: { username, displayName? } }
 */
export async function register(req: RegisterRequest): Promise<LoginResponse> {
    return await postJson<LoginResponse>(api, "/api/auth/register", req);
}

const api = createApiClient({
    // 你后端如果用 cookie/session，把这里改成 true 即可
    withCredentials: false,
});
