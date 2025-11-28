from fastapi import FastAPI

from app.routers.detect import router as detect_router

app = FastAPI(title="Glass Detection API")

# 注册路由
app.include_router(detect_router, prefix="/api/detect")
