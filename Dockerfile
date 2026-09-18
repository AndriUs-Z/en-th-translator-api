FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render กำหนด port ผ่าน environment variable $PORT อัตโนมัติ (ปกติคือ 10000)
ENV PORT=10000
EXPOSE 10000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
