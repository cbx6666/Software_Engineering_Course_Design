from fastapi import FastAPI
from app.routers.detect import router

app = FastAPI(title="Glass Detection API")

# 注册路由
app.include_router(router, prefix="/api/detect")
