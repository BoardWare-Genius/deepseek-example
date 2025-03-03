FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# 安裝 Python 和 pip
# Install Python
RUN apt-get update
RUN apt-get install -y python3-pip python3-venv
RUN pythonm -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY ./requirements.txt /app/requirements.txt
RUN pip install -Ur requirements.txt
# 設置工作目錄
WORKDIR /app

# 複製項目文件
COPY . /app

# 設置環境變量
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# 設置默認命令
EXPOSE 9999
CMD ["python3", "app.py"]