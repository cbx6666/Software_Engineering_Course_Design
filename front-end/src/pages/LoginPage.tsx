import { useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Eye, EyeOff, Loader2, LogIn, Sparkles, UserPlus } from "lucide-react";

export function LoginPage({
    onLogin,
    onRegister,
    isSubmitting,
}: {
    onLogin: (params: { email: string; password: string }) => Promise<void>;
    onRegister: (params: { email: string; password: string }) => Promise<void>;
    isSubmitting: boolean;
}) {
    const [mode, setMode] = useState<"login" | "register">("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const canSubmit = useMemo(() => {
        if (!email.trim() || !password || isSubmitting) return false;
        if (mode === "register") {
            if (confirmPassword && confirmPassword !== password) return false;
        }
        return true;
    }, [email, password, confirmPassword, isSubmitting, mode]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        try {
            if (mode === "login") {
                await onLogin({ email, password });
            } else {
                if (password !== confirmPassword) {
                    setError("两次输入的密码不一致");
                    return;
                }
                await onRegister({ email, password });
            }
        } catch (err) {
            const msg = (mode === "login" ? "登录失败，请稍后再试。" : "注册失败，请稍后再试。");
            setError(msg);
        }
    };

    const header = (
        <div className="mb-6 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/15 backdrop-blur-md">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-slate-200">玻璃智能检测系统</span>
            </div>
            <h1 className="mt-4 bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
                {mode === "login" ? "欢迎登录" : "创建账号"}
            </h1>
            <p className="mt-2 text-slate-300 text-sm">{mode === "login" ? "登录后进入检测主页" : "注册后将自动登录"}</p>
        </div>
    );

    const modeSwitch = (
        <div className="flex items-center justify-end">
            <button
                type="button"
                className="text-sm text-white hover:text-white/90 hover:underline underline-offset-4 transition-colors"
                disabled={isSubmitting}
                onClick={() => {
                    setMode((m) => (m === "login" ? "register" : "login"));
                    setError(null);
                }}
            >
                {mode === "login" ? "没有账号？去注册" : "已有账号？去登录"}
            </button>
        </div>
    );

    const form = (
        <form onSubmit={handleSubmit} className="space-y-3 flex-1 w-full">
            <div className="space-y-2">
                <label className="text-slate-200 text-sm">邮箱</label>
                <Input
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    type="email"
                    inputMode="email"
                    placeholder="请输入邮箱地址"
                    autoComplete="email"
                    disabled={isSubmitting}
                    className="h-9 bg-white/5 border-white/15 text-white placeholder:text-slate-400 focus-visible:ring-cyan-500/30"
                />
            </div>

            <div className="space-y-2">
                <label className="text-slate-200 text-sm">密码</label>

                <div className="relative flex items-center">
                    <Input
                        value={password}
                        placeholder="请输入密码"
                        onChange={(e) => setPassword(e.target.value)}
                        type={showPassword ? "text" : "password"}
                        className="h-9 w-full pr-10 bg-white/5 border-white/15 text-white placeholder:text-slate-400 focus-visible:ring-cyan-500/30"
                    />

                    <button
                        type="button"
                        className="absolute right-2 flex items-center justify-center"
                        style={{ top: "50%", transform: "translateY(-50%)" }}
                        onClick={() => setShowPassword((v) => !v)}
                        aria-label={showPassword ? "隐藏密码" : "显示密码"}
                        disabled={isSubmitting}
                    >
                        {showPassword ? (
                            <EyeOff className="w-4 h-4 text-slate-300" />
                        ) : (
                            <Eye className="w-4 h-4 text-slate-300" />
                        )}
                    </button>
                </div>
            </div>

            {mode === "register" && (
                <div className="space-y-2">
                    <label className="text-slate-200 text-sm">确认密码</label>
                    <Input
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        type={showPassword ? "text" : "password"}
                        placeholder="再次输入密码"
                        autoComplete="new-password"
                        disabled={isSubmitting}
                        className="h-9 bg-white/5 border-white/15 text-white placeholder:text-slate-400 focus-visible:ring-cyan-500/30"
                    />
                    {confirmPassword && confirmPassword !== password && (
                        <div className="text-sm text-red-400">两次输入的密码不一致</div>
                    )}
                </div>
            )}

            {modeSwitch}

            {error && (
                <div className="text-sm text-red-400 bg-red-500/15 px-4 py-3">
                    {error}
                </div>
            )}

            <Button
                type="submit"
                disabled={!canSubmit}
                className="w-full relative overflow-hidden bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0 shadow-lg shadow-cyan-500/25"
                size="lg"
            >
                <span className="relative inline-flex items-center">
                    {isSubmitting ? (
                        <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            {mode === "login" ? "登录中..." : "注册中..."}
                        </>
                    ) : mode === "login" ? (
                        <>
                            <LogIn className="w-4 h-4 mr-2" />
                            登录
                        </>
                    ) : (
                        <>
                            <UserPlus className="w-4 h-4 mr-2" />
                            注册
                        </>
                    )}
                </span>
            </Button>
        </form>
    );

    const card = (
        <div
            className="relative mx-auto"
            style={{
                width: "80%",
                maxWidth: "36rem",
                minHeight: "80vh",
            }}
        >
            <div
                className="absolute -inset-1 rounded-3xl opacity-70 blur-xl"
                style={{
                    background: "linear-gradient(135deg, rgba(34,211,238,0.35), rgba(99,102,241,0.30), rgba(236,72,153,0.24))",
                }}
            />

            <Card className="relative h-full border-0 bg-white/10 backdrop-blur-xl shadow-2xl rounded-3xl overflow-hidden">
                {/* 顶部条纹（移除亮光拂过动效，保持简洁） */}
                <div className="h-1 w-full bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />

                <div className="p-4 pt-3 flex flex-col">{form}</div>
            </Card>
        </div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative overflow-hidden">
            {/* Glass Pattern Background */}
            <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0" style={{
                    backgroundImage: `
                        linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px),
                        linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px)
                    `,
                    backgroundSize: '60px 60px'
                }}></div>
            </div>

            {/* Animated Glass Shards */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-20 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute bottom-20 right-20 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
                <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
            </div>

            <div
                className="relative z-10 min-h-screen flex items-start justify-center p-6"
                style={{ paddingTop: "10rem" }}
            >
                <div className="w-full">
                    {header}
                    {card}
                </div>
            </div>
        </div>
    );
}
