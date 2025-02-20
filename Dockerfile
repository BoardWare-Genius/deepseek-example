FROM python
WORKDIR /app
RUN pip install transformers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY ./models/deepseek-1.5b ./models/deepseek-1.5b
COPY ./gpu_streaming_thread.py ./main.py
CMD ["python", "main.py"]