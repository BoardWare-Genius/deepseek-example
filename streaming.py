import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 替换为你的本地模型路径
model_path = "./models/deepseek-model-8b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)


def generate_streaming_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 初始化生成的文本
    generated_text = prompt

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)

        # 获取下一个词的logits
        next_token_logits = outputs.logits[:, -1, :]

        # 选择概率最高的词
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # 将生成的词添加到输入中
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # 解码生成的词
        next_token = tokenizer.decode(
            next_token_id[0], skip_special_tokens=True)

        # 输出生成的词
        print(next_token, end=" ", flush=True)

        # 更新生成的文本
        generated_text += next_token

        # 如果生成结束符，则停止生成
        if next_token_id == tokenizer.eos_token_id:
            break

    return generated_text


# 示例调用
prompt = "你是誰"
generated_text = generate_streaming_text(prompt)
print()
