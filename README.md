# EN-TH Translator API

FastAPI backend สำหรับแปลภาษาอังกฤษ → ไทย ด้วยโมเดล RNN-augmented Transformer
(quantize เป็น int8 แล้ว) — ใช้ Greedy Decode พร้อม repetition guard

## ไฟล์ที่ต้องเพิ่มก่อนใช้งาน

นำไฟล์ 2 ไฟล์นี้จาก Colab Notebook มาวางไว้ในโฟลเดอร์เดียวกับ `app.py`:

- `sp_en_th.model` (จาก Cell 4)
- `model_quantized.onnx` (จาก Cell 13.5)

## รันทดสอบบนเครื่องตัวเอง

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

ทดสอบที่ `http://localhost:8000/docs`

## Endpoint

**POST /translate**

Request:
```json
{ "text": "Machine learning is a subset of artificial intelligence." }
```

Response:
```json
{
  "original": "Machine learning is a subset of artificial intelligence.",
  "translated": "..."
}
```

**GET /health** — เช็คว่า server พร้อมใช้งาน

## Deploy ขึ้น Render

ดูขั้นตอนละเอียดใน `DEPLOY_GUIDE.md`
