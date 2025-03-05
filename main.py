import uvicorn
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from chat_model import ChatModel
print(f"version:{torch.__version__}, cuda:{torch.cuda.is_available()}")
chat_model = ChatModel()
app = FastAPI()


class Prompt(BaseModel):
    text: str


async def sse(prompt, max_length: int = 100):
    next_token_func = await chat_model.start_chat(prompt, max_length)
    count = 0
    while True:
        text = await next_token_func()
        if text is None:
            break
        yield f"data: {text}\n\n"
        if count >= max_length:
            break
        count += 1


def sync_sse(prompt, max_length: int = 100):
    next_token_func = chat_model.sync_start_chat(prompt, max_length)
    count = 0
    while True:
        text = next_token_func()
        if text is None:
            break
        yield f"data: {text}\n\n"
        if count >= max_length:
            break
        count += 1


@app.post("/")
async def root(prompt: Prompt):
    return StreamingResponse(sse(prompt.text), media_type="text/event-stream")


@app.post("/{wildcard}")
async def root(prompt: Prompt):
    return StreamingResponse(sse(prompt.text), media_type="text/event-stream")


@app.post("/sync")
async def root(prompt: Prompt):
    return StreamingResponse(sync_sse(prompt.text), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
