FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
  software-properties-common \
  build-essential apt-utils \
  wget curl vim git ca-certificates kmod \
  nvidia-driver-525 \
  && rm -rf /var/lib/apt/lists/*

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
  python3.12.2 \
  python3.12.2-dev \
  python3.12.2-distutils \
  python3.12.2-lib2to3 \
  python3.12.2-gdbm \
  python3.12.2-tk \
  pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12.2 999 \
  && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

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