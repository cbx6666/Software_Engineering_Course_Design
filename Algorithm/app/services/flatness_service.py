import json
import base64
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, Literal
from contextlib import contextmanager

from fastapi import HTTPException, UploadFile

from algorithms.flatness_detection.detect import main as run_flatness_detection

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatnessService:
    """封装平整度流程：保存输入、调用管道、返回前端所需结构。
    
    安全改进：
    - 使用临时目录存储用户上传的文件和处理结果
    - 每个请求使用独立的会话目录，避免文件冲突
    - 处理完成后自动清理临时文件
    - 文件不在 web 可访问路径下
    """

    def __init__(self):
        # 使用系统临时目录，不在项目目录下，更安全
        self.temp_base = Path(tempfile.gettempdir()) / "glass_detection"
        self.temp_base.mkdir(parents=True, exist_ok=True)
        # 设置目录权限（仅所有者可访问）
        if hasattr(self.temp_base, 'chmod'):
            try:
                self.temp_base.chmod(0o700)
            except Exception:
                pass  # Windows 可能不支持 chmod

    @contextmanager
    def _create_session_dirs(self):
        """创建会话临时目录，使用上下文管理器确保自动清理"""
        session_id = str(uuid.uuid4())
        session_dir = self.temp_base / session_id
        data_dir = session_dir / "data"
        result_dir = session_dir / "result"
        
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            # 设置目录权限
            if hasattr(data_dir, 'chmod'):
                try:
                    data_dir.chmod(0o700)
                    result_dir.chmod(0o700)
                except Exception:
                    pass
            
            yield data_dir, result_dir
        finally:
            # 自动清理整个会话目录
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir, ignore_errors=True)
                except Exception as e:
                    print(f"清理临时目录失败: {e}")

    def _ensure_dirs(self, data_dir: Path, result_dir: Path):
        """确保数据目录和结果目录存在"""
        data_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

    async def _save_upload(self, file: UploadFile, target_stem: str, data_dir: Path) -> Path:
        """保存上传的文件到数据目录"""
        suffix = Path(file.filename or "").suffix.lower()
        if not suffix:
            suffix = ".jpg"
        if suffix not in ACCEPTED_EXTS:
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 格式不支持")

        dest = data_dir / f"{target_stem}{suffix}"
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
            {
                "label": "RMS 偏差 (mm)",
                "value": f"{flat_rms:.2f}",
                "description": "均方根偏差（Root Mean Square），反映所有测量点相对于理想平面的整体偏差水平。RMS 考虑了所有点的贡献，对异常值敏感度适中，是工业上常用的平整度评估指标。\n\n计算方法：RMS = √(Σ(d_i - d_mean)² / N)，其中 d_i 为各点偏差，d_mean 为平均偏差，N 为点数。"
            },
            {
                "label": "95%分位数 (mm)",
                "value": f"{flat_p95:.2f}",
                "description": "95%分位数表示有95%的测量点偏差小于此值，忽略5%的极端值。该指标对异常值不敏感，能更稳健地反映整体平整度水平，适合评估存在局部缺陷的情况。"
            },
            {
                "label": "平整度范围 (mm)",
                "value": f"{flat_range:.2f}",
                "description": "最大偏差与最小偏差的差值，反映整个测量区域的偏差跨度。范围值大说明表面起伏明显，但无法反映偏差的分布特征。"
            },
            {
                "label": "平均偏差 (mm)",
                "value": f"{metrics.get('flatness_mean_mm', 0.0):.2f}",
                "description": "所有测量点偏差的平均值。接近0表示整体无系统性偏差，正值表示整体偏高，负值表示整体偏低。"
            },
            {
                "label": "标准差 (mm)",
                "value": f"{metrics.get('flatness_std_mm', 0.0):.2f}",
                "description": "偏差值的标准差，反映偏差的离散程度。标准差大说明偏差分布分散，平整度一致性差。"
            },
            {
                "label": "最大偏差 (mm)",
                "value": f"{metrics.get('max_mm', 0.0):.2f}",
                "description": "所有测量点中的最大正偏差值，表示表面最高点相对于理想平面的高度。"
            },
            {
                "label": "最小偏差 (mm)",
                "value": f"{metrics.get('min_mm', 0.0):.2f}",
                "description": "所有测量点中的最小偏差值（通常为负值），表示表面最低点相对于理想平面的深度。"
            },
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
                {
                    "label": "错误详情",
                    "value": (details or message)[:2000],
                    "description": "检测过程中发生的错误信息。请检查输入文件格式、算法参数设置或系统日志。"
                },
            ],
        }

    async def run_flatness(
        self,
        left_env: UploadFile,
        left_mix: UploadFile,
        right_env: UploadFile,
        right_mix: UploadFile,
    ) -> Dict:
        """保存四张输入图并执行完整平整度管道，返回 DetectionResultData 结构。
        
        使用临时目录和会话隔离，处理完成后自动清理。
        """
        # 使用会话临时目录，自动清理
        with self._create_session_dirs() as (data_dir, result_dir):
            # 保存上传的文件
            await self._save_upload(left_env, "left_env", data_dir)
            await self._save_upload(left_mix, "left_mix", data_dir)
            await self._save_upload(right_env, "right_env", data_dir)
            await self._save_upload(right_mix, "right_mix", data_dir)

            # 执行平整度检测流程，传入自定义目录
            try:
                run_flatness_detection(
                    data_dir=str(data_dir),
                    result_dir=str(result_dir)
                )
            except HTTPException:
                raise
            except Exception as exc:
                return self._build_failure_result("平整度流程执行失败", str(exc), repr(exc))

            # 读取平整度指标
            metrics_path = result_dir / "flatness_metrics.json"
            if not metrics_path.exists():
                return self._build_failure_result("未生成平整度指标文件", "请检查角点检测与平整度重建日志。")

            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return self._build_failure_result("平整度结果解析失败", str(exc))
        
            response = self._build_response(metrics)

            # 读取并附加 3D 点云数据（包含投影点）
            pc_data_path = result_dir / "pointcloud_data.json"
            if pc_data_path.exists():
                try:
                    pc_data = json.loads(pc_data_path.read_text(encoding="utf-8"))
                    response["pointcloud"] = pc_data
                except Exception as e:
                    print(f"读取点云数据失败: {e}")
        
            # 在结果目录中寻找 flatness.png 并附加为 data URI
            try:
                flatness_image_path = result_dir / "flatness.png"
                if flatness_image_path.exists():
                    img_bytes = flatness_image_path.read_bytes()
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    response["image"] = f"data:image/png;base64,{b64}"
            except Exception:
                # 附加失败不影响主流程
                pass

            return response