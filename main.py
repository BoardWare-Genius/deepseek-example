from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定本地路径
model_path = "./models/deepseek-model-8b"

# 从本地加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

input_text = "你好，DeepSeek模型"

# 编码输入文本
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成文本
output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=2  # 生成 2 个不同的结果
)

# 解码并打印生成的文本
for i, sequence in enumerate(output):
    print(
        f"Generated text {i+1}: {tokenizer.decode(sequence, skip_special_tokens=True)}")
