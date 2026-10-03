import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI


# =========================
# 基础配置
# =========================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="CareerCopilot API",
    version="1.0.0",
    description="AI 求职与岗位匹配助手后端"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# SiliconFlow API
# =========================

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

if not SILICONFLOW_API_KEY:
    raise ValueError("没有读取到 SILICONFLOW_API_KEY")

client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)


# =========================
# 数据模型
# =========================

class CareerRequest(BaseModel):
    resume: str
    job_description: str


# =========================
# 首页
# =========================

@app.get("/")
def root():
    index_file = BASE_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "CareerCopilot API 正常运行",
        "status": "success"
    }


# =========================
# AI 岗位分析接口
# =========================

@app.post("/api/analyze")
def analyze(request: CareerRequest):

    prompt = f"""
你是一名专业的AI职业求职顾问。

请根据用户简历和目标岗位JD进行分析。

【用户简历】
{request.resume}

【目标岗位】
{request.job_description}

请按照以下结构输出：

### 岗位匹配度分析

#### 1. 岗位匹配度
分析用户简历与岗位要求的整体匹配情况。

#### 2. 简历中的匹配优势
列出用户已经具备、并且与岗位要求直接相关的能力。

#### 3. 当前存在的能力差距
指出用户目前还需要提升的地方。

#### 4. JD要求但简历中缺失的技能
指出岗位要求中，用户简历没有明显体现的内容。

#### 5. 简历优化建议
给出具体、可执行的简历修改建议。

#### 6. 面试准备建议
给出与该岗位相关的面试准备建议。

最后给出一个岗位匹配度评分，格式：

### 岗位匹配度评分：XX/100

请根据实际简历内容进行判断，不要编造用户没有提供的经历。
"""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {
                    "role": "system",
                    "content": "你是一名专业的AI职业求职顾问。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )

        result = response.choices[0].message.content

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
