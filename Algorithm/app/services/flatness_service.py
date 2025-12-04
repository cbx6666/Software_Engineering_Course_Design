import json
from pathlib import Path
from typing import Dict, Literal

from fastapi import HTTPException, UploadFile

from algorithms.flatness_detection.flatness_pipeline import run_pipeline as pipeline_runner

import base64

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatnessService:
    """封装平整度流程：保存输入、调用管道、返回前端所需结构。"""

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

    def _build_response(self, metrics: Dict[str, float]) -> Dict:
        flat_range = metrics.get("flatness_range_mm", 0.0)
        if flat_range <= 3.0:
            status: Literal["success", "warning", "error"] = "success"
            title = "平整度正常"
            description = "测得的平整度范围处于优良阈值内。"
        elif flat_range <= 5.0:
            status = "warning"
            title = "平整度存在轻微偏差"
            description = "平整度超过优良阈值，建议进一步复核。"
        else:
            status = "error"
            title = "平整度超出安全范围"
            description = "检测到较大平整度偏差，请安排检修。"

        details = [
            {"label": "平整度范围 (mm)", "value": f"{flat_range:.2f}"},
            {"label": "RMS 偏差 (mm)", "value": f"{metrics.get('flatness_rms_mm', 0.0):.2f}"},
            {"label": "平均偏差 (mm)", "value": f"{metrics.get('flatness_mean_mm', 0.0):.2f}"},
            {"label": "标准差 (mm)", "value": f"{metrics.get('flatness_std_mm', 0.0):.2f}"},
            {"label": "最大偏差 (mm)", "value": f"{metrics.get('max_mm', 0.0):.2f}"},
            {"label": "最小偏差 (mm)", "value": f"{metrics.get('min_mm', 0.0):.2f}"},
        ]

        return {
            "status": status,
            "title": title,
            "description": description,
            "details": details,
        }

    def _build_failure_result(self, title: str, message: str, details: str | None = None) -> Dict:
        return {
            "status": "error",
            "title": title,
            "description": message,
            "details": [
                {"label": "错误详情", "value": (details or message)[:2000]},
            ],
        }

    async def run_flatness(
        self,
        left_env: UploadFile,
        left_mix: UploadFile,
        right_env: UploadFile,
        right_mix: UploadFile,
    ) -> Dict:
        """保存四张输入图并执行完整平整度管道，返回 DetectionResultData 结构。"""
        self.pipeline.ensure_dir(self.pipeline.PIPELINE_DATA)
        self.pipeline.clear_directory(self.pipeline.PIPELINE_DATA)

        await self._save_upload(left_env, "left_env")
        await self._save_upload(left_mix, "left_mix")
        await self._save_upload(right_env, "right_env")
        await self._save_upload(right_mix, "right_mix")

        try:
            self.pipeline.main()
        except HTTPException:
            raise
        except Exception as exc:
            return self._build_failure_result("平整度流程执行失败", str(exc), repr(exc))

        metrics_path = self.pipeline.POINTCLOUD_DIR / "result" / "flatness_metrics.json"
        if not metrics_path.exists():
            return self._build_failure_result("未生成平整度指标文件", "请检查角点检测与平整度重建日志。")

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._build_failure_result("平整度结果解析失败", str(exc))
    
        response = self._build_response(metrics)
    
        # 在指定算法输出目录中寻找 pointcloud.png 并附加为 data URI
        try:
            alg_result_dir = Path(__file__).resolve().parents[2] / "algorithms" / "flatness_detection" / "pointcloud_gen" / "result"
            candidate = alg_result_dir / "pointcloud.png"
            if candidate.exists():
                img_bytes = candidate.read_bytes()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                response["image"] = f"data:image/png;base64,{b64}"
        except Exception:
            # 附加失败不影响主流程
            pass

        return response