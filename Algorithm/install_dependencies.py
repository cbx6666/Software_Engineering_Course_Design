import subprocess
import sys

def install_dependencies():
    """自动安装requirements.txt中的依赖库"""
    try:
        # 检查pip是否可用
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("错误：未找到pip，请先安装Python并配置环境变量")
        return

    try:
        # 安装依赖
        print("开始安装依赖库...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("所有依赖安装完成！")
    except subprocess.CalledProcessError as e:
        print(f"安装失败：{e}")
    except FileNotFoundError:
        print("错误：未找到requirements.txt，请确保文件在当前目录")

if __name__ == "__main__":
    install_dependencies()