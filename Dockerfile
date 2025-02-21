FROM python:3.12.2 AS builder
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
