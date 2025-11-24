from openai import OpenAI
import openai
client = OpenAI()

usage = openai.api_usage()

# Output the usage details
print(usage)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo", #"gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "1 + 1 = ?."
        }
    ]
)

print(completion.choices[0].message)
# import os
# import openai

# # 从环境变量中读取 API Key
# api_key = os.getenv('OPENAI_API_KEY')

# if not api_key:
#     raise ValueError("请设置 OPENAI_API_KEY 环境变量")

# # 设置 API Key
# openai.api_key = api_key

# 现在你可以使用 openai 库调用 API 了