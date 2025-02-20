import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 检查GPU是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 替换为实际的DeepSeek模型名称或路径
model_name = "./models/deepseek-model-8b"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载分类模型并移动到GPU
model = AutoModelForSequenceClassification.from_pretrained(
    model_name).to(device)

# 输入文本
text = "你是誰"

# 使用分词器对文本进行编码
inputs = tokenizer(text, return_tensors="pt").to(device)

# 使用模型进行推理
outputs = model(**inputs)

# 输出分类结果
print(outputs)
