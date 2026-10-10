import os
import re
import math
import unicodedata
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sentencepiece as spm

# ==============================================================================
# 1. การตั้งค่าระบบและ Hyperparameters
# ==============================================================================
app = FastAPI(title="CS English-to-Thai Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
D_MODEL = 256
NHEAD = 8
ENC_GRU_HIDDEN = 64
DEC_GRU_HIDDEN = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 128

SPM_MODEL_PATH = os.getenv("SPM_MODEL_PATH", "sp_en_th.model")
MODEL_PATH = (
    "finetuned_model.pt"
    if os.path.exists("finetuned_model.pt")
    else "best_model.pt"
)

# ==============================================================================
# 2. โครงสร้างสถาปัตยกรรมโมเดล (RNN-Augmented Transformer)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RNNAugmentedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, bidirectional=True, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.norm1(x + self.dropout(gru_out))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class RNNAugmentedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, bidirectional=False, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.norm1(x + self.dropout(gru_out))
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_out))
        cross_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        x = self.norm3(x + self.dropout(cross_out))
        ff_out = self.ff(x)
        x = self.norm4(x + self.dropout(ff_out))
        return x

class RNNAugmentedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD,
                 enc_gru_hidden=ENC_GRU_HIDDEN, dec_gru_hidden=DEC_GRU_HIDDEN,
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, max_seq_len=MAX_LEN, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([
            RNNAugmentedEncoderLayer(d_model, nhead, enc_gru_hidden, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            RNNAugmentedDecoderLayer(d_model, nhead, dec_gru_hidden, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.embed_scale = math.sqrt(d_model)

    def make_padding_mask(self, seq):
        return (seq == self.pad_id)

    def make_causal_mask(self, size, dev):
        return torch.triu(torch.ones(size, size, device=dev), diagonal=1).bool()

    def encode(self, src):
        src_key_padding_mask = self.make_padding_mask(src)
        x = self.embedding(src) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask=None):
        tgt_key_padding_mask = self.make_padding_mask(tgt)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        x = self.embedding(tgt) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        logits = self.output_proj(x)
        return logits

# ==============================================================================
# 3. โหลด Tokenizer และ Model Weights
# ==============================================================================
if not os.path.exists(SPM_MODEL_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ Tokenizer: {SPM_MODEL_PATH}")

sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ Model: {MODEL_PATH}")

model = RNNAugmentedTransformer(vocab_size=sp.get_piece_size()).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"โหลดโมเดลสำเร็จจาก {MODEL_PATH} (Vocab size: {sp.get_piece_size()})")

# ==============================================================================
# 4. ฟังก์ชัน Normalization & Greedy Inference (เร็ว เบา CPU)
# ==============================================================================
def normalize_text_input(text: str) -> str:
    """แปลงเครื่องหมายคำพูดทุกแบบให้เป็น straight quotes และเว้นวรรคให้ token ได้สะอาด"""
    text = unicodedata.normalize("NFKC", text)
    # แปลง “ ” ‘ ’ ` เป็น " หรือ '
    text = re.sub(r'[“”„«»]', '"', text)
    text = re.sub(r"[‘’`´]", "'", text)
    # เติม space รอบ quote เพื่อไม่ให้ติดกับคำด้านใน
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r"'", " ' ", text)
    return re.sub(r'\s+', ' ', text).strip()

def postprocess_thai_output(text: str) -> str:
    """ทำความสะอาดคำแปลภาษาไทย จัดการระยะเครื่องหมายคำพูดให้ชิดคำ"""
    # จัดช่องว่างรอบ quote ให้เป็นระเบียบ ไม่ติดคำจนอ่านไม่ออก
    text = re.sub(r'\s*"\s*([^"]+?)\s*"\s*', r' "\1" ', text)
    text = re.sub(r"\s*'\s*([^']+?)\s*'\s*", r" '\1' ", text)
    return re.sub(r'\s+', ' ', text).strip()

def encode_text(text: str):
    clean_text = normalize_text_input(text)
    ids = sp.encode(clean_text, out_type=int)
    ids = [BOS_ID] + ids + [EOS_ID]
    return ids[:MAX_LEN]

def decode_ids(ids):
    clean_ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    raw_decoded = sp.decode(clean_ids)
    return postprocess_thai_output(raw_decoded)

def get_banned_ngram_tokens(generated_ids, ngram_size=3):
    if len(generated_ids) < ngram_size - 1:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i:i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned

@torch.no_grad()
def greedy_translate(src_ids, max_new_tokens=120, repetition_penalty=1.3):
    """Fast Greedy Search แบบดั้งเดิม (เร็วที่สุดบน CPU)"""
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, memory_key_padding_mask = model.encode(src)

    safe_max_len = min(max_new_tokens, MAX_LEN - 1)
    tgt_ids = [BOS_ID]

    for _ in range(safe_max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token_logits = logits[0, -1, :].clone()

        # Multiplicative Repetition Penalty
        for tok in set(tgt_ids):
            if next_token_logits[tok] > 0:
                next_token_logits[tok] /= repetition_penalty
            else:
                next_token_logits[tok] *= repetition_penalty

        # Strict 3-gram Blocking กันพูดซ้ำ
        banned = get_banned_ngram_tokens(tgt_ids, ngram_size=3)
        for tok in banned:
            next_token_logits[tok] = float("-inf")

        next_token = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token)

        if next_token == EOS_ID:
            break

    return tgt_ids

def translate_sentence(text: str) -> str:
    text_clean = text.strip()
    if not text_clean:
        return ""
    src_ids = encode_text(text_clean)
    out_ids = greedy_translate(src_ids)
    return decode_ids(out_ids)

def split_document_into_sentences(text: str):
    paragraphs = text.split("\n")
    units = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            units.append(("", True))
            continue
        raw_sentences = re.split(r'(?<=[.?!;])\s+', p_clean)
        for s in raw_sentences:
            if s.strip():
                units.append((s.strip(), False))
        units.append(("", True))
    return units

def translate_document(document_text: str) -> str:
    units = split_document_into_sentences(document_text)
    translated_result = []
    for sent, is_newline in units:
        if is_newline:
            translated_result.append("\n")
        else:
            th_sent = translate_sentence(sent)
            translated_result.append(th_sent + " ")
    return "".join(translated_result).strip()

# ==============================================================================
# 5. FastAPI Endpoints
# ==============================================================================
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    original: str
    translated: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL_PATH,
        "device": str(device)
    }

@app.post("/translate", response_model=TranslationResponse)
def api_translate_sentence(req: TranslationRequest):
    """แปลประโยคเดี่ยวสั้นๆ รวดเร็วด้วย Greedy Search"""
    try:
        translated = translate_sentence(req.text)
        return TranslationResponse(original=req.text, translated=translated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate-document", response_model=TranslationResponse)
def api_translate_document(req: TranslationRequest):
    """แปลเอกสาร/ข้อความยาวทีละประโยค"""
    try:
        translated = translate_document(req.text)
        return TranslationResponse(original=req.text, translated=translated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
