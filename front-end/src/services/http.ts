import axios, { AxiosError, type AxiosInstance, type AxiosRequestConfig } from "axios";

export type BackendMessageShape =
    | string
    | {
        message?: unknown;
        error?: unknown;
    };

export function getBaseURL(): string {
    // 允许：
    // - 不配置：走同源（例如 Vite 反代 / 后端同域部署）
    // - 配置：VITE_BACKEND_URL=http://localhost:8000 这类
    const raw = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? "";
    return raw.replace(/\/+$/, "");
}

export function createApiClient(options?: {
    baseURL?: string;
    withCredentials?: boolean;
    timeout?: number;
}): AxiosInstance {
    return axios.create({
        baseURL: options?.baseURL ?? getBaseURL(),
        withCredentials: options?.withCredentials ?? false,
        timeout: options?.timeout ?? 60000,
    });
}

export function normalizeAxiosError(e: unknown): Error {
    if (!axios.isAxiosError(e)) {
        return e instanceof Error ? e : new Error("请求失败，请稍后重试");
    }

    const err = e as AxiosError<BackendMessageShape>;

    // 后端如果返回字符串/JSON message，尽量透传
    const data = err.response?.data;
    const backendMsg =
        typeof data === "string"
            ? data
            : typeof (data as any)?.message === "string"
                ? (data as any).message
                : typeof (data as any)?.error === "string"
                    ? (data as any).error
                    : undefined;

    if (backendMsg) return new Error(backendMsg);

    if (err.code === "ECONNABORTED") return new Error("请求超时，请检查网络或稍后重试");
    if (!err.response) return new Error("无法连接到服务器，请检查后端是否启动/地址是否正确");

    const status = err.response.status;
    return new Error(`请求失败（HTTP ${status}）`);
}

export async function postJson<T>(
    client: AxiosInstance,
    path: string,
    body: unknown,
    config?: AxiosRequestConfig,
): Promise<T> {
    try {
        const { data } = await client.post<T>(path, body, {
            headers: { "Content-Type": "application/json" },
            ...config,
        });
        return data;
    } catch (e) {
        throw normalizeAxiosError(e);
    }
}

export async function postFormData<T>(
    client: AxiosInstance,
    path: string,
    formData: FormData,
    config?: AxiosRequestConfig,
): Promise<T> {
    try {
        const { data } = await client.post<T>(path, formData, {
            headers: { "Content-Type": "multipart/form-data" },
            ...config,
        });
        return data;
    } catch (e) {
        throw normalizeAxiosError(e);
    }
}


