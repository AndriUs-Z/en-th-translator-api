import os
import onnxruntime as ort
import sentencepiece as spm
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Token IDs ต้องตรงกับตอน train (ดูจาก Cell 4 ใน Notebook) ---
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MAX_LEN = 128  # ต้องตรงกับ MAX_LEN ตอน export ONNX เป๊ะ (fixed-length export)
NO_REPEAT_LAST_N = 6
REPETITION_PENALTY_VALUE = 8.0

# --- โหลด tokenizer และโมเดลที่ quantize แล้วตอน server เริ่มทำงานครั้งเดียว ---
print("Loading tokenizer and model...")
sp = spm.SentencePieceProcessor(model_file="sp_en_th.model")
session = ort.InferenceSession("model_quantized.onnx", providers=["CPUExecutionProvider"])
print("Loaded successfully!")

app = FastAPI(title="EN-TH Translator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เปิดกว้างไว้ก่อน ค่อยจำกัดทีหลังให้เหลือแค่ origin ของระบบหลัก
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    original: str
    translated: str


def encode_and_pad(text: str, max_len: int = MAX_LEN):
    """
    แปลงข้อความเป็น token ids แล้ว pad ให้ยาวเท่ากับ MAX_LEN เสมอ
    ต้องตรงกับความยาวตอน export ONNX เป๊ะ เพราะโมเดลถูก export แบบ fixed-length
    (nn.MultiheadAttention จำความยาว sequence ตอน trace เป็นค่าคงที่ในกราฟ ONNX)
    """
    ids = [BOS_ID] + sp.encode(text, out_type=int)[:max_len - 2] + [EOS_ID]
    real_len = len(ids)
    ids = ids + [PAD_ID] * (max_len - real_len)
    return ids, real_len


def decode_ids(ids):
    filtered = [i for i in ids if i not in (BOS_ID, EOS_ID, PAD_ID)]
    return sp.decode(filtered)


def greedy_translate_onnx(text: str):
    """
    Greedy decode ผ่าน ONNX Runtime พร้อม repetition guard
    ทุก input (src, tgt) ต้อง pad ให้ยาว MAX_LEN คงที่เสมอ เพราะโมเดล export
    แบบ fixed-length (ดู encode_and_pad ด้านบน)
    """
    src_ids, _ = encode_and_pad(text)
    src = np.array([src_ids], dtype=np.int64)

    tgt_ids = [BOS_ID]  # token จริงที่ generate ไปแล้ว (ไม่รวม padding)

    for _ in range(MAX_LEN - 1):
        # pad tgt_ids ปัจจุบันให้ยาว MAX_LEN ก่อนส่งเข้าโมเดลทุกครั้ง
        current_len = len(tgt_ids)
        tgt_padded = tgt_ids + [PAD_ID] * (MAX_LEN - current_len)
        tgt = np.array([tgt_padded], dtype=np.int64)

        logits = session.run(["logits"], {"src": src, "tgt": tgt})[0]
        # ตำแหน่งที่ต้องดู logits คือ token จริงตัวสุดท้าย (ก่อนโดน pad)
        next_token_logits = logits[0, current_len - 1, :].copy()

        # repetition guard: ลด logit ของ token ที่เพิ่งออกไปเมื่อไม่นานมานี้ตรง ๆ
        recent = tgt_ids[-NO_REPEAT_LAST_N:] if len(tgt_ids) >= NO_REPEAT_LAST_N else tgt_ids
        for tok in set(recent):
            next_token_logits[tok] -= REPETITION_PENALTY_VALUE

        next_token = int(np.argmax(next_token_logits))
        tgt_ids.append(next_token)

        if next_token == EOS_ID or len(tgt_ids) >= MAX_LEN:
            break

    return tgt_ids


@app.get("/")
def root():
    return {"status": "ok", "message": "EN-TH Translator API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        out_ids = greedy_translate_onnx(req.text)
        translated = decode_ids(out_ids)
        return {"original": req.text, "translated": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"translate failed: {e}")
