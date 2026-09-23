from openai import OpenAI

client = OpenAI(
    api_key="sk-e97c561fb4ab42ed9b5e07453f181c84",
    base_url="https://api.deepseek.com"
)

text = """Video emotion analysis from VLM:
dominant_emotion: sadness
valence: -0.7
arousal: 0.6
description: Anna is crying and looks sad in a dark cave."""

prompt = f"""任务：请根据以下“视频情绪分析描述”推测视频中表达的情绪强度。
对高兴、惊讶、悲伤、愤怒、厌恶、恐惧六种情绪类别进行评价，评分为0到7之间的连续取值（可精确到小数点后一位），0表示完全没有，7表示非常强烈。

输入（描述）：
{text}

请严格按以下格式给出评分结果（无需返回额外评论）：
高兴: [评分]; 惊讶: [评分]; 悲伤: [评分]; 愤怒: [评分]; 厌恶: [评分]; 恐惧: [评分]
"""

resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

print(resp.choices[0].message.content)