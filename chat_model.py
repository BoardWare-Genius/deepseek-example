import asyncio

import torch
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel:

    def __init__(self, model_path: str = "./model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path).to(self.device)

    def sync_start_chat(self, prompt, max_length: int):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.device)  # 将输入数据移动到 GPU
        input_ids = inputs["input_ids"]
        count = 0

        def next_token():
            nonlocal count
            if count >= max_length:
                return None
            nonlocal input_ids
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1).unsqueeze(0)
            if next_token_id == self.tokenizer.eos_token_id:
                return None
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            next_token = self.tokenizer.decode(
                next_token_id[0], skip_special_tokens=True)
            count += 1
            return next_token
        return next_token

    async def start_chat(self, prompt, max_length: int):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.device)  # 将输入数据移动到 GPU
        input_ids = inputs["input_ids"]
        loop = asyncio.get_running_loop()
        count = 0

        async def next_token():
            nonlocal count
            if count >= max_length:
                return None
            nonlocal input_ids
            outputs = await loop.run_in_executor(None, self.model, input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(
                next_token_logits, dim=-1).unsqueeze(0)
            if next_token_id == self.tokenizer.eos_token_id:
                return None
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            next_token = self.tokenizer.decode(
                next_token_id[0], skip_special_tokens=True)
            count += 1
            return next_token
        return next_token
