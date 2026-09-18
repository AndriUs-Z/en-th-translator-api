# คู่มือ Deploy โมเดลแปลภาษาขึ้น Render

คู่มือนี้สอนวิธีนำไฟล์ที่ได้จาก Notebook (`sp_en_th.model`, `model_quantized.onnx`)
ไป deploy เป็น API สาธารณะบน Render เพื่อให้เว็บแอปพลิเคชันหลัก (Frontend/Backend)
เรียกใช้งานได้

> **หมายเหตุ:** เดิมคู่มือนี้แนะนำ Hugging Face Spaces แต่ตั้งแต่กรกฎาคม 2026
> Hugging Face เปลี่ยนนโยบายให้ Docker SDK และ Gradio SDK ต้องใช้ Plan แบบ
> เสียเงิน (PRO) เหลือแค่ Static Space ที่ยังฟรี จึงเปลี่ยนมาใช้ **Render**
> แทน เพราะยังมี free tier ที่รองรับ Docker อยู่ ไม่ต้องใช้บัตรเครดิต

---

## สิ่งที่ต้องเตรียมก่อน

จาก Notebook ที่ train เสร็จแล้ว ดาวน์โหลดไฟล์ 2 ไฟล์นี้มาเก็บไว้:

| ไฟล์ | มาจาก Cell ไหน |
|---|---|
| `sp_en_th.model` | Cell 4 (SentencePiece Tokenizer) |
| `model_quantized.onnx` | Cell 13.5 (Quantization) |

และไฟล์โค้ดสำหรับรัน API ที่แนบมาในโฟลเดอร์นี้:

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | โค้ด FastAPI ที่โหลดโมเดลและเปิด endpoint `/translate` |
| `requirements.txt` | รายชื่อ library ที่ต้องติดตั้ง |
| `Dockerfile` | คำสั่ง build container สำหรับ Render |
| `README.md` | คำอธิบายการใช้งานเบื้องต้น |

---

## ขั้นตอนที่ 1 — สร้าง Repository บน GitHub

Render deploy จาก GitHub repo เป็นหลัก จึงต้องมี repo ก่อน:

1. สร้างบัญชี [github.com](https://github.com) (ถ้ายังไม่มี)
2. สร้าง repository ใหม่ ตั้งชื่อเช่น `en-th-translator-api`
3. เลือก **Private** ก็ได้ (Render เชื่อมต่อ private repo ได้)

---

## ขั้นตอนที่ 2 — เตรียมโฟลเดอร์สำหรับอัปโหลด

รวมไฟล์ทั้งหมดให้อยู่ในโฟลเดอร์เดียวกัน:

```
en-th-translator-api/
├── app.py
├── requirements.txt
├── Dockerfile
├── sp_en_th.model          ← ไฟล์ที่ดาวน์โหลดจาก Colab
└── model_quantized.onnx    ← ไฟล์ที่ดาวน์โหลดจาก Colab
```

นำ `sp_en_th.model` และ `model_quantized.onnx` ที่ดาวน์โหลดจาก Colab
มาวางไว้ในโฟลเดอร์เดียวกับ `app.py` ตามโครงสร้างด้านบน

> **ขนาดไฟล์:** ถ้า `model_quantized.onnx` มีขนาดใหญ่กว่า 100MB ต้องใช้
> [Git LFS](https://git-lfs.github.com) ในการ push ขึ้น GitHub (`git lfs install`
> แล้ว `git lfs track "*.onnx"` ก่อน commit)

---

## ขั้นตอนที่ 3 — Push ขึ้น GitHub

```bash
cd en-th-translator-api
git init

# ถ้าไฟล์ .onnx ใหญ่กว่า 100MB ให้ track ด้วย git lfs ก่อน
git lfs install
git lfs track "*.onnx"
git add .gitattributes

git add .
git commit -m "initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/en-th-translator-api.git
git push -u origin main
```

---

## ขั้นตอนที่ 4 — สร้าง Web Service บน Render

1. สมัคร/login ที่ [render.com](https://render.com) ด้วยบัญชี GitHub (ไม่ต้องใส่บัตรเครดิต)
2. กด **New** → **Web Service**
3. เลือก repo `en-th-translator-api` ที่เพิ่ง push ไป
4. ตั้งค่า:
   - **Name**: `en-th-translator-api`
   - **Runtime**: **Docker** (Render จะ detect `Dockerfile` ให้อัตโนมัติ)
   - **Instance Type**: **Free**
5. กด **Create Web Service**

Render จะ build และ deploy ให้อัตโนมัติ ใช้เวลาประมาณ 3-5 นาที ดู progress ได้ที่แท็บ **Logs**

---

## ขั้นตอนที่ 5 — ทดสอบ API

เมื่อ deploy สำเร็จ Render จะให้ URL รูปแบบ `https://en-th-translator-api.onrender.com`

ทดสอบได้ที่:

```
https://en-th-translator-api.onrender.com/docs
```

จะเห็นหน้า Swagger UI ให้ทดสอบ endpoint `POST /translate` ได้ทันที ลองส่ง:

```json
{
  "text": "Machine learning is a subset of artificial intelligence."
}
```

> **หมายเหตุ:** Free tier ของ Render จะ sleep หลังไม่มีคนเรียกใช้ 15 นาที
> การเรียกครั้งแรกหลัง sleep จะช้ากว่าปกติ (cold start ประมาณ 30-60 วินาที)
> ครั้งถัดไปจะเร็วปกติจนกว่าจะ sleep อีกครั้ง

---

## ขั้นตอนที่ 6 — จำกัด CORS ให้ปลอดภัยขึ้น (แนะนำก่อนใช้งานจริง)

ใน `app.py` ตอนนี้เปิด CORS ให้ทุก origin เรียกได้ (`allow_origins=["*"]`)
เมื่อรู้ domain ของเว็บแอปพลิเคชันหลักแล้ว ควรแก้ให้จำกัดเฉพาะ origin นั้น:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-main-app-domain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

แล้ว commit และ push ขึ้น GitHub อีกครั้ง — Render จะ auto-deploy เวอร์ชันใหม่ให้เอง
ทุกครั้งที่ push (auto-deploy เปิดอยู่โดย default)

---

## สรุป URL ที่ได้

หลัง deploy สำเร็จ จะได้ URL รูปแบบนี้สำหรับให้ระบบหลักเรียกใช้:

```
https://en-th-translator-api.onrender.com/translate
```

เก็บ URL นี้ไว้ใช้เชื่อมกับ Backend/Frontend ของระบบหลักต่อไป
