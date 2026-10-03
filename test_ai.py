import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

if not api_key:
    raise ValueError("没有读取到 SILICONFLOW_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[
        {
            "role": "system",
            "content": "你是 CareerCopilot 的 AI 求职助手。"
        },
        {
            "role": "user",
            "content": "你好，请用一句话介绍你自己。"
        }
    ],
    temperature=0.7
)

print("\n========== CareerCopilot AI ==========\n")
print(response.choices[0].message.content)
print("\n=======================================\n")