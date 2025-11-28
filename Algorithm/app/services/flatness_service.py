import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException, UploadFile

from algorithms.flatness_detection.flatness_pipeline import run_pipeline as pipeline_runner

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatnessService:
    """封装平整度流程：保存输入、调用管道、返回指标。"""

    def __init__(self):
        self.pipeline = pipeline_runner

    async def _save_upload(self, file: UploadFile, target_stem: str) -> Path:
        suffix = Path(file.filename or "").suffix.lower()
        if not suffix:
            suffix = ".jpg"
        if suffix not in ACCEPTED_EXTS:
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 格式不支持")

        dest = self.pipeline.PIPELINE_DATA / f"{target_stem}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = await file.read()
        dest.write_bytes(data)
        return dest

    async def run_flatness(
        self,
        left_env: UploadFile,
        left_mix: UploadFile,
        right_env: UploadFile,
        right_mix: UploadFile,
    ) -> Optional[Dict[str, float]]:
        """保存四张输入图并执行完整平整度管道，返回指标。"""
        self.pipeline.ensure_dir(self.pipeline.PIPELINE_DATA)
        self.pipeline.clear_directory(self.pipeline.PIPELINE_DATA)

        await self._save_upload(left_env, "left_env")
        await self._save_upload(left_mix, "left_mix")
        await self._save_upload(right_env, "right_env")
        await self._save_upload(right_mix, "right_mix")

        try:
            self.pipeline.main()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"平整度流程执行失败: {exc}") from exc

        metrics_path = self.pipeline.POINTCLOUD_DIR / "result" / "flatness_metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        return None