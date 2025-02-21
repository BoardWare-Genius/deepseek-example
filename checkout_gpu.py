import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


@app.get("/")
async def root():
    return f"version:{torch.__version__}, cuda:{torch.cuda.is_available()}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
