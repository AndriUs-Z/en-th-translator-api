import os
import math
import torch
import torch.nn as nn
import sentencepiece as spm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CS Translation Service")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# 1. นิยามโครงสร้างโมเดล (ตรงกับ Cell 6 ใน Notebook)
# ---------------------------------------------------------------------------
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MAX_LEN = 128
D_MODEL = 256
NHEAD = 8
ENC_GRU_HIDDEN = 64
DEC_GRU_HIDDEN = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

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
            nn.Linear(dim_feedforward, d_model),
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
            nn.Linear(dim_feedforward, d_model),
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

    def make_causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

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
        return self.output_proj(x)

# ---------------------------------------------------------------------------
# 2. โหลด Tokenizer และ Model Weights
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPM_PATH = "sp_en_th.model"
MODEL_PATH = "finetuned_model.pt" if os.path.exists("finetuned_model.pt") else "best_model.pt"

if not os.path.exists(SPM_PATH):
    raise FileNotFoundError(f"Missing {SPM_PATH}")

sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
model = RNNAugmentedTransformer(vocab_size=sp.get_piece_size()).to(device)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded weights from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Missing {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 3. Translation Logic (จาก Cell 11)
# ---------------------------------------------------------------------------
def encode_text(text: str):
    ids = sp.encode(text, out_type=int)
    return ([BOS_ID] + ids + [EOS_ID])[:MAX_LEN]

def decode_ids(ids):
    clean_ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(clean_ids)

@torch.inference_mode()
def greedy_translate(src_ids, max_new_tokens=150, no_repeat_last_n=3, repetition_penalty_value=5.0):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, memory_key_padding_mask = model.encode(src)
    safe_max_len = min(max_new_tokens, MAX_LEN - 1)

    tgt_ids = [BOS_ID]
    for _ in range(safe_max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token_logits = logits[0, -1, :].clone()

        recent = tgt_ids[-no_repeat_last_n:] if len(tgt_ids) >= no_repeat_last_n else tgt_ids
        for tok in set(recent):
            next_token_logits[tok] -= repetition_penalty_value

        next_token = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token)
        if next_token == EOS_ID:
            break

    return decode_ids(tgt_ids)

# ---------------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(req: TranslationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    src_ids = encode_text(req.text)
    translated = greedy_translate(src_ids)
    return TranslationResponse(translated_text=translated)