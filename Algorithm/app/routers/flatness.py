import json
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from algorithms.flatness_pipeline import run_pipeline as pipeline_runner

router = APIRouter(prefix="/flatness", tags=["flatness"])

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


async def _save_upload(file: UploadFile, target_stem: str) -> Path:
    """
    保存上传的图片到 flatness_pipeline/data 下，并返回保存路径。
    target_stem: 期望的文件前缀（left_env 等）。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if not suffix:
        suffix = ".jpg"
    if suffix not in ACCEPTED_EXTS:
        raise HTTPException(status_code=400, detail=f"文件 {file.filename} 格式不支持")

    dest = pipeline_runner.PIPELINE_DATA / f"{target_stem}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    return dest


@router.post("/run")
async def run_flatness_pipeline(
    left_env: UploadFile = File(...),
    left_mix: UploadFile = File(...),
    right_env: UploadFile = File(...),
    right_mix: UploadFile = File(...),
):
    """
    接收左右相机的 env/mix 图像，写入 flatness_pipeline/data，并执行完整平整度流程。
    返回流程状态以及最终的平整度指标（如果生成）。
    """
    pipeline_runner.ensure_dir(pipeline_runner.PIPELINE_DATA)
    pipeline_runner.clear_directory(pipeline_runner.PIPELINE_DATA)

    await _save_upload(left_env, "left_env")
    await _save_upload(left_mix, "left_mix")
    await _save_upload(right_env, "right_env")
    await _save_upload(right_mix, "right_mix")

    try:
        pipeline_runner.main()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"平整度流程执行失败: {exc}") from exc

    metrics_path = pipeline_runner.POINTCLOUD_DIR / "result" / "flatness_metrics.json"
    metrics: Dict[str, float] | None = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "status": "success",
        "message": "平整度流程执行完成",
        "metrics": metrics,
    }

