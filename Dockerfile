FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# 安裝 Python 和 pip
# Install Python
RUN apt-get update
RUN apt-get install -y python3-pip

# 設置工作目錄
WORKDIR /app

# 複製項目文件
COPY . /app

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r  requirements.txt

# 設置環境變量
ENV PYTHONUNBUFFERED=1

# 設置默認命令
EXPOSE 9999
CMD ["python3", "app.py"]