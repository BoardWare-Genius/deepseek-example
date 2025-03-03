FROM python:3.12.2 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 第二阶段：最终镜像
FROM python:3.12.2-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

RUN apt-get update && \
  apt-get install -y --no-install-recommends some-package && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /root/.cache/pip
EXPOSE 9999
CMD ["python", "app.py"]