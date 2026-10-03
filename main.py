import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

# =========================
# 1. 初始化 FastAPI
# =========================

app = FastAPI(
    title="CareerCopilot API",
    description="AI 求职与岗位匹配助手后端",
    version="1.0.0"
)

# =========================
# 2. 配置跨域
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3. 初始化 SiliconFlow
# =========================

api_key = os.getenv("SILICONFLOW_API_KEY")

if not api_key:
    raise ValueError("没有读取到 SILICONFLOW_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# =========================
# 4. 请求数据结构
# =========================

class CareerRequest(BaseModel):
    resume: str
    job_description: str


# =========================
# 5. 首页测试
# =========================

@app.get("/")
def root():
    return {
        "message": "CareerCopilot API 正常运行",
        "status": "success"
    }


# =========================
# 6. AI 岗位匹配
# =========================

@app.post("/api/analyze")
def analyze(request: CareerRequest):

    prompt = f"""
你是一名专业的AI招聘与职业发展助手 CareerCopilot。

请根据用户简历和目标岗位JD，对两者进行分析。

【用户简历】
{request.resume}

【目标岗位JD】
{request.job_description}

请从以下几个方面进行分析：

1. 岗位匹配度
2. 简历中的匹配优势
3. 当前存在的能力差距
4. JD要求但简历中缺失的技能
5. 简历优化建议
6. 面试准备建议

请使用清晰、专业、结构化的中文回答。

最后给出一个0-100之间的岗位匹配度评分。
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "你是 CareerCopilot，一名专业的AI求职助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    result = response.choices[0].message.content

    return {
        "success": True,
        "result": result
    }