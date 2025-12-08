from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.detect import router as detect_router

app = FastAPI(title="Glass Detection API")

# 配置 CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # 前端开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(detect_router, prefix="/api/detect")
