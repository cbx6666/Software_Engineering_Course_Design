from fastapi import FastAPI
from app.routers import detect

app = FastAPI(title="Glass Detection API")

# 注册路由
app.include_router(detect.router, prefix="/api/detect")
