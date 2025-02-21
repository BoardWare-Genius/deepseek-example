import asyncio
import json
from typing import Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from chat_model import ChatModel

chat_model = ChatModel()
app = FastAPI()


class Prompt(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


async def sse(prompt, max_length: int = 100):
    next_token_func = await chat_model.start_chat(prompt, max_length)
    count = 0
    while True:
        text = await next_token_func()
        if text is None:
            break
        yield f"event: next\ndata: {count} {text} \n\n"
        if max_length >= 100:
            break
        count += 1


@app.post("/")
async def root(prompt: Prompt):
    return StreamingResponse(sse(prompt), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
