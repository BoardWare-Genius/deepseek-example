import asyncio
import contextvars
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# 替换为你的本地模型路径
model_path = "./models/deepseek-model-8b"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path).to(device)  # 将模型移动到 GPU

canceled_context = contextvars.ContextVar("canceled", default=False)


async def generate_streaming_text_with_callback(prompt, callback, max_length=100):
    if callback is None:
        return
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # 将输入数据移动到 GPU
    input_ids = inputs["input_ids"]

    # 初始化生成的文本
    generated_text = prompt
    loop = asyncio.get_running_loop()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = await loop.run_in_executor(None, model, input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            next_token = tokenizer.decode(
                next_token_id[0], skip_special_tokens=True)

            callback(next_token)
            generated_text += next_token
            if next_token_id == tokenizer.eos_token_id:
                break
    return generated_text


async def generate_streaming_with_yield(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # 将输入数据移动到 GPU
    input_ids = inputs["input_ids"]
    loop = asyncio.get_running_loop()
    yield "next_token"
    with torch.no_grad():
        for _ in range(max_length):
            outputs = await loop.run_in_executor(None, model, input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            next_token = tokenizer.decode(
                next_token_id[0], skip_special_tokens=True)
            yield next_token
            if next_token_id == tokenizer.eos_token_id:
                break

if __name__ == "__main__":
    for text in generate_streaming_with_yield("你是誰"):
        print(text)
