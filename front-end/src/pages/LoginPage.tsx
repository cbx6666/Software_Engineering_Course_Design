import { useMemo, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { GlassInspectionVisual } from "@/components/visual/GlassInspectionVisual";
import {
    Eye,
    EyeOff,
    Loader2,
    LockKeyhole,
    LogIn,
    Mail,
    ShieldCheck,
    UserPlus,
} from "lucide-react";

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

    const handleSubmit = async (e: FormEvent) => {
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
        } catch {
            const msg = mode === "login" ? "登录失败，请稍后再试。" : "注册失败，请稍后再试。";
            setError(msg);
        }
    };

    return (
        <div className="login-page">
            <div className="shell-scanline" />
            <div className="login-shell">
                <section className="login-visual">
                    <div className="eyebrow">
                        <ShieldCheck />
                        QC GLASS LAB
                    </div>
                    <h1>玻璃检测控制台</h1>
                    <p>
                        登录后进入裂纹检测、幕墙平整度分析和历史复核流程。界面为现场质检任务优化，突出上传、判断、点云查看和追溯。
                    </p>
                    <div className="tag-row">
                        {["裂纹定位", "平整度点云", "检测台账"].map((tag) => (
                            <span className="tag-pill" key={tag}>{tag}</span>
                        ))}
                    </div>
                    <div className="login-visual__graphic">
                        <GlassInspectionVisual />
                    </div>
                </section>

                <section className="login-card">
                    <div className="login-card__head">
                        <div className="eyebrow">
                            {mode === "login" ? <LogIn /> : <UserPlus />}
                            {mode === "login" ? "ACCOUNT ACCESS" : "NEW OPERATOR"}
                        </div>
                        <h2>{mode === "login" ? "欢迎登录" : "创建账号"}</h2>
                        <p>{mode === "login" ? "使用账号进入玻璃检测工作台。" : "注册后将自动进入检测系统。"}</p>
                    </div>

                    <form onSubmit={handleSubmit} className="login-form">
                        <div className="form-field">
                            <label>邮箱</label>
                            <div className="password-field">
                                <Mail className="field-icon" />
                                <input
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    type="email"
                                    inputMode="email"
                                    placeholder="请输入邮箱地址"
                                    autoComplete="email"
                                    disabled={isSubmitting}
                                    className="lab-input lab-input--icon"
                                />
                            </div>
                        </div>

                        <div className="form-field">
                            <label>密码</label>
                            <div className="password-field">
                                <LockKeyhole className="field-icon" />
                                <input
                                    value={password}
                                    placeholder="请输入密码"
                                    type={showPassword ? "text" : "password"}
                                    onChange={(e) => setPassword(e.target.value)}
                                    autoComplete={mode === "login" ? "current-password" : "new-password"}
                                    disabled={isSubmitting}
                                    className="lab-input lab-input--icon"
                                />
                                <button
                                    type="button"
                                    className="password-toggle"
                                    onClick={() => setShowPassword((visible) => !visible)}
                                    aria-label={showPassword ? "隐藏密码" : "显示密码"}
                                >
                                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                </button>
                            </div>
                        </div>

                        {mode === "register" && (
                            <div className="form-field">
                                <label>确认密码</label>
                                <input
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    type={showPassword ? "text" : "password"}
                                    placeholder="再次输入密码"
                                    autoComplete="new-password"
                                    disabled={isSubmitting}
                                    className="lab-input"
                                />
                                {confirmPassword && confirmPassword !== password && (
                                    <div className="form-error">两次输入的密码不一致</div>
                                )}
                            </div>
                        )}

                        <button
                            type="button"
                            className="mode-switch"
                            disabled={isSubmitting}
                            onClick={() => {
                                setMode((m) => (m === "login" ? "register" : "login"));
                                setError(null);
                            }}
                        >
                            {mode === "login" ? "没有账号？去注册" : "已有账号？去登录"}
                        </button>

                        {error && <div className="form-error">{error}</div>}

                        <Button
                            type="submit"
                            disabled={!canSubmit}
                            className="lab-primary-button"
                            size="lg"
                        >
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
                        </Button>
                    </form>
                </section>
            </div>
        </div>
    );
}
