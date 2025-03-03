FROM nvidia/cuda:12.8.0-base-oraclelinux8

# 安裝 Python 和 pip
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製項目文件
COPY . /app

# 安裝 Python 依賴
RUN pip3 install -r requirements.txt

# 設置環境變量
ENV PYTHONUNBUFFERED=1

# 設置默認命令
EXPOSE 9999
CMD ["python3", "app.py"]