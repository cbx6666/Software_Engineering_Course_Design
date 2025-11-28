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
    
    except Exception:
        return {
            "status": "error", 
            "title": "图片转换为数据流失败",
            "description": None,
            "details": None
        }

@router.post("/glass-flatness")
async def detect_glass_flatness(
    left_env: UploadFile = File(...),
    left_mix: UploadFile = File(...),
    right_env: UploadFile = File(...),
    right_mix: UploadFile = File(...),
):
    return await flatness_service.run_flatness(left_env, left_mix, right_env, right_mix)