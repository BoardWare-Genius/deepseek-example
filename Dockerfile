FROM python:3.12.2 AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY . .

FROM python:3.12.2-slim
COPY --from=builder /app /app
WORKDIR /app
CMD ["python", "main.py"]