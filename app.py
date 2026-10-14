import gc
import math
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sentencepiece as spm
import torch
import torch.nn as nn

# ==============================================================================
# 1. การตั้งค่า Resource (จำกัด Thread เซฟ RAM)
# ==============================================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cpu")

# ==============================================================================
# 2. Hyperparameters & File Paths
# ==============================================================================
D_MODEL = 256
NHEAD = 8
ENC_GRU_HIDDEN = 64
DEC_GRU_HIDDEN = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 128

MAX_NEW_TOKENS = 64
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.2

SPM_EN_TH_PATH = os.getenv("SPM_EN_TH_PATH", "sp_en_th.model")
SPM_TH_EN_PATH = os.getenv("SPM_TH_EN_PATH", "sp_th_en.model")

# เปลี่ยนชื่อไฟล์ให้ตรงกับไฟล์ที่บีบอัดแล้ว (8-bit)
MODEL_EN_TH_PATH = os.getenv("MODEL_EN_TH_PATH", "q_finetuned_model.pt")
MODEL_TH_EN_PATH = os.getenv("MODEL_TH_EN_PATH", "best_model_th_en.pt")

# ==============================================================================
# 3. โหลด Tokenizer แยก 2 ตัว
# ==============================================================================
if not os.path.exists(SPM_EN_TH_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์: {SPM_EN_TH_PATH}")
if not os.path.exists(SPM_TH_EN_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์: {SPM_TH_EN_PATH}")

print("[*] กำลังโหลด SentencePiece Tokenizers...")
sp_en_th = spm.SentencePieceProcessor(model_file=SPM_EN_TH_PATH)
sp_th_en = spm.SentencePieceProcessor(model_file=SPM_TH_EN_PATH)

vocab_size_en_th = sp_en_th.get_piece_size()
vocab_size_th_en = sp_th_en.get_piece_size()

PAD_ID = sp_en_th.pad_id() if sp_en_th.pad_id() != -1 else 0
BOS_ID = sp_en_th.bos_id() if sp_en_th.bos_id() != -1 else 2
EOS_ID = sp_en_th.eos_id() if sp_en_th.eos_id() != -1 else 3

# ==============================================================================
# 4. Model Classes
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
    def forward(self, x): return self.dropout(x + self.pe[:, : x.size(1), :])

class RNNAugmentedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, bidirectional=True, batch_first=True)
        self.gru_proj = nn.Linear(gru_hidden * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_key_padding_mask=None):
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.norm1(x + self.dropout(gru_out))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm3(x + self.dropout(ff_out))

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
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
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
        return self.norm4(x + self.dropout(ff_out))

class RNNAugmentedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, nhead=NHEAD, enc_gru_hidden=ENC_GRU_HIDDEN, dec_gru_hidden=DEC_GRU_HIDDEN, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, max_seq_len=MAX_LEN, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([RNNAugmentedEncoderLayer(d_model, nhead, enc_gru_hidden, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([RNNAugmentedDecoderLayer(d_model, nhead, dec_gru_hidden, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.embed_scale = math.sqrt(d_model)
    def make_padding_mask(self, seq): return seq == self.pad_id
    def make_causal_mask(self, size, dev): return torch.triu(torch.ones(size, size, device=dev), diagonal=1).bool()
    def encode(self, src):
        src_key_padding_mask = self.make_padding_mask(src)
        x = self.embedding(src) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.encoder_layers: x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x, src_key_padding_mask
    def decode(self, tgt, memory, memory_key_padding_mask=None):
        tgt_key_padding_mask = self.make_padding_mask(tgt)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        x = self.embedding(tgt) * self.embed_scale
        x = self.pos_encoding(x)
        for layer in self.decoder_layers: x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.output_proj(x)

# ==============================================================================
# 5. โหลด Checkpoints ที่ผ่านการบีบอัด (Quantized) มาแล้ว
# ==============================================================================
def load_quantized_checkpoint(model_path, target_vocab_size):
    if not os.path.exists(model_path):
        print(f"[!] ไม่พบไฟล์ {model_path}")
        return None
        
    # สร้างโครงสร้างโมเดลเปล่า
    model = RNNAugmentedTransformer(vocab_size=target_vocab_size).to(device)
    model.eval()
    
    # **สำคัญมาก:** ต้องจำลองการบีบอัดโครงสร้างก่อน เพื่อให้มันรับกับน้ำหนัก 8-bit ที่จะโหลดเข้ามา
    model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    
    # โหลดน้ำหนัก (state_dict) ที่บีบอัดมาจาก Colab
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # เช็คเผื่อกรณีไฟล์ถูกห่อด้วย key "model_state_dict" (ป้องกัน Error)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    
    model.load_state_dict(state_dict)
    del ckpt, state_dict
    gc.collect()
    
    print(f"[+] โหลดโมเดลบีบอัด {model_path} สำเร็จ (vocab_size={target_vocab_size})")
    return model

print("[*] กำลังโหลดโมเดลเข้าหน่วยความจำ (สามารถโหลด 2 ตัวพร้อมกันได้ เพราะกิน RAM น้อยลง)...")
model_en_th = load_quantized_checkpoint(MODEL_EN_TH_PATH, vocab_size_en_th)
model_th_en = load_quantized_checkpoint(MODEL_TH_EN_PATH, vocab_size_th_en)
print("[+] โหลดเสร็จสิ้น ระบบพร้อมทำงาน")

# ==============================================================================
# 6. ฟังก์ชัน Translation & Chunking
# ==============================================================================
def encode_text(text: str, target_sp, add_bos_eos: bool = True):
    ids = target_sp.encode(text, out_type=int)
    if add_bos_eos: ids = [BOS_ID] + ids + [EOS_ID]
    return ids[:MAX_LEN]

def decode_ids(ids, target_sp):
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return target_sp.decode(ids)

@torch.no_grad()
def greedy_translate(model, src_ids):
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, memory_key_padding_mask = model.encode(src)

    safe_max_len = min(MAX_NEW_TOKENS, MAX_LEN - 1)
    tgt_ids = [BOS_ID]
    seen_tokens = set()

    for _ in range(safe_max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token_logits = logits[0, -1, :].clone()

        if REPETITION_PENALTY != 1.0 and seen_tokens:
            for tok in seen_tokens:
                if next_token_logits[tok] > 0: next_token_logits[tok] /= REPETITION_PENALTY
                else: next_token_logits[tok] *= REPETITION_PENALTY

        next_token = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token)
        seen_tokens.add(next_token)

        if next_token == EOS_ID: break

    return tgt_ids

def translate_long_text(model, text: str, target_sp) -> str:
    if not text or not text.strip(): return ""
