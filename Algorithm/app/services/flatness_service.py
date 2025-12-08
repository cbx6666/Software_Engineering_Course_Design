import json
import base64
from pathlib import Path
from typing import Dict, Literal

from fastapi import HTTPException, UploadFile

from algorithms.flatness_detection.detect import main as run_flatness_detection

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatnessService:
    """封装平整度流程：保存输入、调用管道、返回前端所需结构。"""

    def __init__(self):
        # 获取算法模块的基础路径
        self.base_dir = Path(__file__).resolve().parents[2] / "algorithms" / "flatness_detection"
        self.data_dir = self.base_dir / "data"
        self.result_dir = self.base_dir / "result"

    def _ensure_dirs(self):
        """确保数据目录和结果目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _clear_data_dir(self):
        """清空数据目录"""
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_file():
                    item.unlink()

    async def _save_upload(self, file: UploadFile, target_stem: str) -> Path:
        """保存上传的文件到数据目录"""
        suffix = Path(file.filename or "").suffix.lower()
        if not suffix:
            suffix = ".jpg"
        if suffix not in ACCEPTED_EXTS:
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 格式不支持")

        dest = self.data_dir / f"{target_stem}{suffix}"
        data = await file.read()
        dest.write_bytes(data)
        return dest

    def _build_response(self, metrics: Dict[str, float]) -> Dict:
        # 使用 RMS 作为主要判断标准（更稳健，考虑所有点）
        # RMS 是工业上常用的平整度评估指标，能反映整体偏差分布
        flat_rms = metrics.get("flatness_rms_mm", 0.0)
        
        # 辅助判断：95%分位数（忽略5%极端值，更稳健）
        flat_p95 = metrics.get("p95_mm", 0.0)
        
        # 参考指标：范围（用于了解极端情况，但不作为主要判断）
        flat_range = metrics.get("flatness_range_mm", 0.0)
        
        # 主要使用 RMS 判断，同时检查是否有极端值（p95 远大于 RMS）
        # 阈值设置（可根据实际工程标准调整）
        # 如果 p95 远大于 RMS，说明存在局部极端偏差，需要额外关注
        has_extreme_values = flat_p95 > flat_rms * 2.5
        
        if flat_rms <= 10.0:
            status: Literal["success", "warning", "error"] = "success"
            title = "平整度正常"
            if has_extreme_values:
                description = f"RMS偏差 {flat_rms:.2f} mm 正常，但存在局部极端偏差（95%分位数 {flat_p95:.2f} mm），建议检查局部区域。"
            else:
                description = f"RMS偏差 {flat_rms:.2f} mm，95%分位数 {flat_p95:.2f} mm，处于优良阈值内。"
        elif flat_rms <= 15.0:
            status = "warning"
            title = "平整度存在轻微偏差"
            description = f"RMS偏差 {flat_rms:.2f} mm，95%分位数 {flat_p95:.2f} mm，超过优良阈值，建议进一步复核。"
        else:
            status = "error"
            title = "平整度超出安全范围"
            description = f"RMS偏差 {flat_rms:.2f} mm，95%分位数 {flat_p95:.2f} mm，检测到较大平整度偏差，请安排检修。"

        details = [
            {"label": "RMS 偏差 (mm) [主要指标]", "value": f"{flat_rms:.2f}"},
            {"label": "95%分位数 (mm) [稳健指标]", "value": f"{flat_p95:.2f}"},
            {"label": "平整度范围 (mm) [参考]", "value": f"{flat_range:.2f}"},
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
        # 确保目录存在并清空数据目录
        self._ensure_dirs()
        self._clear_data_dir()

        # 保存上传的文件
        await self._save_upload(left_env, "left_env")
        await self._save_upload(left_mix, "left_mix")
        await self._save_upload(right_env, "right_env")
        await self._save_upload(right_mix, "right_mix")

        # 执行平整度检测流程
        try:
            run_flatness_detection()
        except HTTPException:
            raise
        except Exception as exc:
            return self._build_failure_result("平整度流程执行失败", str(exc), repr(exc))

        # 读取平整度指标
        metrics_path = self.result_dir / "flatness_metrics.json"
        if not metrics_path.exists():
            return self._build_failure_result("未生成平整度指标文件", "请检查角点检测与平整度重建日志。")

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._build_failure_result("平整度结果解析失败", str(exc))
    
        response = self._build_response(metrics)
    
        # 在结果目录中寻找 flatness.png 并附加为 data URI
        try:
            flatness_image_path = self.result_dir / "flatness.png"
            if flatness_image_path.exists():
                img_bytes = flatness_image_path.read_bytes()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                response["image"] = f"data:image/png;base64,{b64}"
        except Exception:
            # 附加失败不影响主流程
            pass

        return response