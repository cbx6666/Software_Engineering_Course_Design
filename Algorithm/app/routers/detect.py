from fastapi import APIRouter, UploadFile, File
from app.services.glass_crack_service import GlassCrackService

router = APIRouter()
crack_service = GlassCrackService()

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

# @router.post("/glass-flatness")
# async def detect_glass_flatness(image: UploadFile = File(...)):
#     result = await flatness_service.detect_flatness(image)
#     return result