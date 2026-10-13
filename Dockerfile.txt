FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง libgomp1 สำหรับ PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render จะ inject ค่า $PORT มาให้เองอัตโนมัติ
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]