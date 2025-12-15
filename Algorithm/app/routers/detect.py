from fastapi import APIRouter, UploadFile, File
from app.services.glass_crack_service import GlassCrackService
from app.services.flatness_service import FlatnessService

router = APIRouter()
crack_service = GlassCrackService()
flatness_service = FlatnessService()

@router.post("/glass-crack")
async def detect_glass_crack(image: UploadFile = File(...)):
    try:
        image_content = await image.read()
        result = crack_service.detect_crack(image_content)
        return result
    
    except Exception as e:
        return {
            "status": "error", 
            "title": "图片转换为数据流失败",
            "description": str(e),
            "details": [
                {
                    "label": "错误详情",
                    "value": str(e)[:2000],
                    "description": "图片处理过程中发生的错误。请检查图片格式是否正确，或联系技术支持。"
                }
            ]
        }

@router.post("/glass-flatness")
async def detect_glass_flatness(
    left_env: UploadFile = File(...),
    left_mix: UploadFile = File(...),
    right_env: UploadFile = File(...),
    right_mix: UploadFile = File(...),
):
    return await flatness_service.run_flatness(left_env, left_mix, right_env, right_mix)