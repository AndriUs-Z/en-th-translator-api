import os
import onnxruntime as ort
import sentencepiece as spm
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Token IDs ต้องตรงกับตอน train (ดูจาก Cell 4 ใน Notebook) ---
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MAX_NEW_TOKENS = 150
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


def encode_text(text: str, max_len: int = 128):
    ids = [BOS_ID] + sp.encode(text, out_type=int)[:max_len - 2] + [EOS_ID]
    return ids


def decode_ids(ids):
    filtered = [i for i in ids if i not in (BOS_ID, EOS_ID, PAD_ID)]
    return sp.decode(filtered)


def greedy_translate_onnx(src_ids):
    """
    Greedy decode ผ่าน ONNX Runtime พร้อม repetition guard
    (ตรงกับ logic ใน Cell 11 ของ Notebook)
    model.onnx export จาก forward(src, tgt) -> logits ทั้งประโยคในครั้งเดียว
    จึงต้องเรียก session.run ใหม่ทุก step โดยขยาย tgt ไปทีละ token
    """
    src = np.array([src_ids], dtype=np.int64)
    tgt_ids = [BOS_ID]

    for _ in range(MAX_NEW_TOKENS):
        tgt = np.array([tgt_ids], dtype=np.int64)
        logits = session.run(["logits"], {"src": src, "tgt": tgt})[0]
        next_token_logits = logits[0, -1, :].copy()

        # repetition guard: ลด logit ของ token ที่เพิ่งออกไปเมื่อไม่นานมานี้ตรง ๆ
        recent = tgt_ids[-NO_REPEAT_LAST_N:] if len(tgt_ids) >= NO_REPEAT_LAST_N else tgt_ids
        for tok in set(recent):
            next_token_logits[tok] -= REPETITION_PENALTY_VALUE

        next_token = int(np.argmax(next_token_logits))
        tgt_ids.append(next_token)

        if next_token == EOS_ID:
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
        src_ids = encode_text(req.text)
        out_ids = greedy_translate_onnx(src_ids)
        translated = decode_ids(out_ids)
        return {"original": req.text, "translated": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"translate failed: {e}")
