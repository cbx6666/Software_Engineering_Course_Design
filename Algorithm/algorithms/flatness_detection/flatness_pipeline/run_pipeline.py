import shutil
import subprocess
from pathlib import Path

# 管道自身的目录结构
PIPELINE_ROOT = Path(__file__).resolve().parent        # flatness_pipeline/
PIPELINE_DATA = PIPELINE_ROOT / "data"                 # 输入 env/mix 图放在这里
PIPELINE_ARTIFACTS = PIPELINE_ROOT / "artifacts"       # 存放中间灰度图、掩膜备份

# 三个子算法所在目录
ALGO_ROOT = PIPELINE_ROOT.parent                       
PROJECTOR_DIR = ALGO_ROOT / "projector_reflection_diff"
STEREO_DIR = ALGO_ROOT / "Stereo_corner_matching"
POINTCLOUD_DIR = ALGO_ROOT / "pointcloud_gen"


def ensure_dir(path: Path) -> Path:
    """
    创建并返回指定目录。
    参数：
      path: 要创建的目录路径。
    说明：
      - parents=True 允许递归创建父目录；
      - exist_ok=True 表示目录已存在时不会报错。
    返回：
      创建后的 Path 对象，方便后续使用。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_directory(path: Path) -> None:
    """
    清空一个目录下的所有文件。
    参数：
      path: 需要被清空的目录 Path。
    说明：
      - 若目录不存在则直接返回；
      - 仅删除文件（entry.is_file），避免误删子目录。
    """
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_file():
            entry.unlink()  # unlink() 相当于删除单个文件


def find_input(prefix: str) -> Path:
    """
    在 pipeline/data 目录中按前缀查找输入图片。
    参数：
      prefix: 文件名前缀，如 "left_env"、"right_mix"。
    说明：
      - 会依次尝试 jpg/jpeg/png/bmp 四种扩展名；
      - 找到第一个存在的文件后返回其 Path；
      - 若所有扩展都找不到，则抛出 FileNotFoundError。
    """
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = PIPELINE_DATA / f"{prefix}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未找到输入文件：{prefix}.[jpg|png|bmp]")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    """执行子模块命令并打印上下文。"""
    print(f"[CMD] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def stage_projector_inputs(camera: str) -> None:
    """将左右相机 env/mix 输入复制到投影差分 data 目录。"""
    camera = camera.lower()
    data_dir = ensure_dir(PROJECTOR_DIR / "data")
    clear_directory(data_dir)

    env_src = find_input(f"{camera}_env")
    mix_src = find_input(f"{camera}_mix")

    env_dst = data_dir / f"env{env_src.suffix.lower()}"
    mix_dst = data_dir / f"mix{mix_src.suffix.lower()}"

    shutil.copy2(env_src, env_dst)
    shutil.copy2(mix_src, mix_dst)
    print(f"[stage] {camera} 环境/混合图已复制至 {data_dir}")


def collect_projector_outputs(camera: str) -> None:
    """把投影差分输出的灰度/掩膜复制到角点模块。"""
    camera = camera.lower()
    result_dir = ensure_dir(PROJECTOR_DIR / "result")

    detect_img = result_dir / "result_detect.png"
    mask_img = result_dir / "result_mask.png"
    if not detect_img.exists() or not mask_img.exists():
        raise FileNotFoundError("未在 projector_reflection_diff/result 下找到输出 result_detect.png/result_mask.png")

    stereo_data = ensure_dir(STEREO_DIR / "data")
    shutil.copy2(detect_img, stereo_data / f"{camera}.png")
    shutil.copy2(mask_img, stereo_data / f"{camera}_mask.png")

    ensure_dir(PIPELINE_ARTIFACTS)
    shutil.copy2(detect_img, PIPELINE_ARTIFACTS / f"{camera}_clean.png")
    shutil.copy2(mask_img, PIPELINE_ARTIFACTS / f"{camera}_mask.png")
    print(f"[stage] {camera} 棋盘灰度/掩膜已复制到 Stereo_corner_matching/data")


def run_projector_stage(camera: str) -> None:
    print(f"\n=== 阶段1：投影差分（{camera}） ===")
    stage_projector_inputs(camera)
    run_cmd(["python", "main.py"], PROJECTOR_DIR)
    collect_projector_outputs(camera)


def run_stereo_stage() -> None:
    print("\n=== 阶段2：角点检测与匹配 ===")
    run_cmd(["python", "main.py"], STEREO_DIR)

    stereo_result = ensure_dir(STEREO_DIR / "result")
    left_json = stereo_result / "corners_left.json"
    right_json = stereo_result / "corners_right.json"
    if not left_json.exists() or not right_json.exists():
        raise FileNotFoundError("未在 Stereo_corner_matching/result 下找到角点 JSON 文件")

    pointcloud_data = ensure_dir(POINTCLOUD_DIR / "data")
    shutil.copy2(left_json, pointcloud_data / "corners_left.json")
    shutil.copy2(right_json, pointcloud_data / "corners_right.json")
    print("[stage] 角点 JSON 已复制到 flatness_detection/pointcloud_gen/data")


def run_flatness_stage() -> None:
    print("\n=== 阶段3：平整度重建 ===")
    run_cmd(["python", "main.py"], POINTCLOUD_DIR)


def main():
    ensure_dir(PIPELINE_DATA)
    ensure_dir(PIPELINE_ARTIFACTS)

    run_projector_stage("left")
    run_projector_stage("right")
    run_stereo_stage()
    run_flatness_stage()
    print("\n 平整度完整流程完成")


if __name__ == "__main__":
    main()

