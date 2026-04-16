from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://172.26.104.240:30033/v1"
)

# 使用你的实际模型路径
model_name = "xverify"
response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
    max_tokens=100
)

print(f"model_name: {model_name} works")

print(response.choices[0].message.content)

print("--------------------------------")