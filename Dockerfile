FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# 安裝 Python 和 pip
# Install Python
RUN apt-get update && \
  apt-get install -y python3-pip python3-dev python-is-python3 && \
  rm -rf /var/lib/apt/lists/*


# 設置工作目錄
WORKDIR /app

# 複製項目文件
COPY . /app

# 安裝 Python 依賴
RUN pip install -r requirements.txt

# 設置環境變量
ENV PYTHONUNBUFFERED=1

# 設置默認命令
EXPOSE 9999
CMD ["python3", "app.py"]