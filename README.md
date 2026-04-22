# Glass Curtain Wall Detection System

## 项目概述

Glass Curtain Wall Detection System 是一个综合性的玻璃幕墙检测平台，旨在通过计算机视觉和深度学习技术自动检测玻璃幕墙的破损和不平整度问题。该系统采用前后端分离架构，结合了先进的图像处理算法和用户友好的Web界面，为建筑质量检测提供高效、准确的解决方案。

## 主要功能

### 🔍 玻璃破损检测 (Glass Crack Detection)

- 基于图像处理的裂缝识别算法
- 支持多种图像格式输入（Base64、本地文件）
- 实时检测和结果可视化
- 高精度特征提取和分类

### 📐 平整度检测 (Flatness Detection)
- 立体视觉技术测量玻璃表面平整度
- 投影反射差分算法
- 点云生成和3D可视化
- 角点检测和匹配算法

### 🌐 Web界面
- 现代化的React前端界面
- 实时检测结果展示
- 3D点云可视化组件
- 响应式设计，支持多种设备

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Front-end     │    │    Back-end     │    │   Algorithm     │
│   (React/Vite)  │◄──►│ (Spring Boot)   │◄──►│   (FastAPI)     │
│                 │    │                 │    │                 │
│ - UI Components │    │ - REST API      │    │ - Crack Detect  │
│ - Image Upload  │    │ - File Upload   │    │ - Flatness Calc │
│ - Result Display│    │ - Data Process  │    │ - Image Process │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 技术栈

- **前端**: React 18, TypeScript, Vite, Tailwind CSS, Radix UI
- **后端**: Java 21, Spring Boot 4.0.0-SNAPSHOT
- **算法**: Python 3.x, FastAPI, OpenCV, NumPy
- **数据库**: (可选，当前版本无数据库依赖)

## 快速开始

### 环境要求

- Node.js 18+
- Java 21+
- Python 3.8+
- Maven 3.6+

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd GlassCurtainWallDetection
   ```

2. **前端安装**
   ```bash
   cd front-end
   npm install
   ```

3. **后端安装**
   ```bash
   cd ../back-end
   mvn clean install
   ```

4. **算法服务安装**
   ```bash
   cd ../algorithm
   pip install -r requirements.txt
   ```

### 运行系统

1. **启动算法服务**
   ```bash
   cd algorithm
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **启动后端服务**
   ```bash
   cd back-end
   mvn spring-boot:run
   ```

3. **启动前端**
   ```bash
   cd front-end
   npm run dev
   ```

4. **访问应用**
   打开浏览器访问 `http://localhost:3000`

### 本地联调默认地址

- 前端: `http://localhost:3000`
- 后端: `http://localhost:8080`
- 算法服务: `http://localhost:8000`

当前仓库默认已按本地联调配置：

- `front-end/.env` 指向 `http://localhost:8080`
- `back-end/src/main/resources/application.properties` 中：
  - `frontend.url=http://localhost:3000`
  - `algorithm.url=http://localhost:8000`

## 使用指南

### 上传图像
1. 在Web界面中选择检测类型（破损检测或平整度检测）
2. 上传玻璃幕墙图像
3. 系统将自动处理并显示检测结果

### 查看结果
- **破损检测**: 显示裂缝位置和严重程度
- **平整度检测**: 提供3D点云可视化和不平整度测量

## 项目结构

```
GlassCurtainWallDetection/
├── algorithm/                 # Python算法服务
│   ├── algorithms/
│   │   ├── crack_detection/   # 裂缝检测算法
│   │   └── flatness_detection/# 平整度检测算法
│   ├── app/                   # FastAPI应用
│   └── requirements.txt
├── back-end/                  # Java Spring Boot后端
│   ├── src/
│   └── pom.xml
├── front-end/                 # React前端
│   ├── src/
│   └── package.json
├── dataset/                   # 数据集和示例
├── docs/                      # 项目文档
│   ├── SRS.pdf               # 软件需求规格说明
│   ├── SDD.pdf               # 软件设计文档
│   └── CPP.pdf               # 课程项目提案
├── literature/               # 相关文献
└── README.md
```

## 数据集

项目使用公开数据集进行训练和测试：

- **玻璃破损数据集**:
  - [Roboflow Glass Defect Dataset](https://universe.roboflow.com/es-h99zz/broken-hmi0f)
  - [Kaggle Glass Defect Dataset](https://www.kaggle.com/datasets/pedrocanoas/glass-defect/data)

## 开发指南

### 代码规范
- 遵循各语言的标准编码规范
- 使用ESLint (前端), Checkstyle (后端), Black (Python)
- 详细规范请参考 `docs/CPP.pdf`

### 测试
```bash
# 前端测试
cd front-end
npm test

# 后端测试
cd back-end
mvn test

# 算法测试
cd algorithm
python -m pytest
```

### 构建生产版本
```bash
# 前端构建
cd front-end
npm run build

# 后端构建
cd back-end
mvn clean package
```

## API文档

### 算法服务 API (FastAPI)
- `POST /api/detect/crack` - 玻璃破损检测
- `POST /api/detect/flatness` - 平整度检测

### 后端 API (Spring Boot)
- RESTful API接口，详情请查看后端代码

## 致谢

感谢所有为这个项目做出贡献的开发者，以及提供数据集的开源社区。
