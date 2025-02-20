from transformers import AutoModel, AutoTokenizer

model_path = "./models/deepseek-model-8b"
# 加载模型和分词器
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 将模型移至 GPU
model.to('cuda')

# 准备输入数据并移至 GPU
inputs = tokenizer("Hello, world!", return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}

text = "这是一个示例文本。"

# 使用分词器对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 使用模型进行推理
outputs = model(**inputs)

# 输出分类结果
print(outputs.logits)
