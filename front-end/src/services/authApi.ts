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
 * 登录接口（占位）
 * - 你后面接入后端时：把 `mockLogin` 换成下面注释的真实请求即可。
 */
export async function login(req: LoginRequest): Promise<LoginResponse> {
    // TODO: 接入后端后启用：
    // import axios from "axios";
    // const baseURL = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/+$/, "") ?? "";
    // const api = axios.create({ baseURL, withCredentials: true });
    // const { data } = await api.post<LoginResponse>("/api/auth/login", req);
    // return data;

    return await mockLogin(req);
}

/**
 * 注册接口（占位）
 * - 后续你接入后端：改为 `POST /api/auth/register`，返回同样的 token/user 即可。
 */
export async function register(req: RegisterRequest): Promise<LoginResponse> {
    // TODO: 接入后端后启用：
    // import axios from "axios";
    // const baseURL = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/+$/, "") ?? "";
    // const api = axios.create({ baseURL, withCredentials: true });
    // const { data } = await api.post<LoginResponse>("/api/auth/register", req);
    // return data;

    return await mockRegister(req);
}

async function mockLogin(req: LoginRequest): Promise<LoginResponse> {
    await new Promise((r) => setTimeout(r, 700));

    // 极简占位：允许任意非空账号/密码
    const email = req.username.trim();
    if (!email || !req.password) throw new Error("请输入邮箱与密码");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) throw new Error("邮箱格式不正确");

    return {
        token: `dev-token:${email}:${Date.now()}`,
        user: {
            username: email,
            displayName: email,
        },
    };
}

async function mockRegister(req: RegisterRequest): Promise<LoginResponse> {
    await new Promise((r) => setTimeout(r, 900));

    const email = req.username.trim();
    if (!email) throw new Error("请输入邮箱");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) throw new Error("邮箱格式不正确");
    if (!req.password) throw new Error("请输入密码");
    if (req.password.length < 6) throw new Error("密码至少 6 位");

    return {
        token: `dev-token:${email}:${Date.now()}`,
        user: {
            username: email,
            displayName: req.displayName?.trim() || email,
        },
    };
}


