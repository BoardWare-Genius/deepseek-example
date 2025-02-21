FROM python
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY . .
CMD ["python", "main.py"]