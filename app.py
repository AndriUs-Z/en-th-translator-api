import gc
import math
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sentencepiece as spm
import torch
import torch.nn as nn

# ==============================================================================
# 1. การตั้งค่า Resource (Render Free Tier 512MB RAM)
# ==============================================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cpu")

# ==============================================================================
# 2. Hyperparameters & Architecture (ตาม Colab Notebook)
# ==============================================================================
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

MAX_NEW_TOKENS = 128
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.2

SPM_MODEL_PATH = os.getenv("SPM_MODEL_PATH", "sp_en_th.model")
MODEL_EN_TH_PATH = os.getenv("MODEL_EN_TH_PATH", "finetuned_model.pt")
MODEL_TH_EN_PATH = os.getenv("MODEL_TH_EN_PATH", "finetuned_model_th_en.pt")


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x):
    return self.dropout(x + self.pe[:, : x.size(1), :])


class RNNAugmentedEncoderLayer(nn.Module):

  def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
    super().__init__()
    self.gru = nn.GRU(d_model, gru_hidden, bidirectional=True, batch_first=True)
    self.gru_proj = nn.Linear(gru_hidden * 2, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.self_attn = nn.MultiheadAttention(
        d_model, nhead, dropout=dropout, batch_first=True
    )
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
    attn_out, _ = self.self_attn(
        x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False
    )
    x = self.norm2(x + self.dropout(attn_out))
    ff_out = self.ff(x)
    return self.norm3(x + self.dropout(ff_out))


class RNNAugmentedDecoderLayer(nn.Module):

  def __init__(self, d_model, nhead, gru_hidden, dim_feedforward, dropout):
    super().__init__()
    self.gru = nn.GRU(
        d_model, gru_hidden, bidirectional=False, batch_first=True
    )
    self.gru_proj = nn.Linear(gru_hidden, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.self_attn = nn.MultiheadAttention(
        d_model, nhead, dropout=dropout, batch_first=True
    )
    self.norm2 = nn.LayerNorm(d_model)
    self.cross_attn = nn.MultiheadAttention(
        d_model, nhead, dropout=dropout, batch_first=True
    )
    self.norm3 = nn.LayerNorm(d_model)
    self.ff = nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )
    self.norm4 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(
      self,
      x,
      memory,
      tgt_mask=None,
      tgt_key_padding_mask=None,
      memory_key_padding_mask=None,
  ):
    gru_out, _ = self.gru(x)
    gru_out = self.gru_proj(gru_out)
    x = self.norm1(x + self.dropout(gru_out))
    attn_out, _ = self.self_attn(
        x,
        x,
        x,
        attn_mask=tgt_mask,
        key_padding_mask=tgt_key_padding_mask,
        need_weights=False,
    )
    x = self.norm2(x + self.dropout(attn_out))
    cross_out, _ = self.cross_attn(
        x,
        memory,
        memory,
        key_padding_mask=memory_key_padding_mask,
        need_weights=False,
    )
    x = self.norm3(x + self.dropout(cross_out))
    ff_out = self.ff(x)
    return self.norm4(x + self.dropout(ff_out))


class RNNAugmentedTransformer(nn.Module):

  def __init__(
      self,
      vocab_size,
      d_model=D_MODEL,
      nhead=NHEAD,
      enc_gru_hidden=ENC_GRU_HIDDEN,
      dec_gru_hidden=DEC_GRU_HIDDEN,
      num_encoder_layers=NUM_ENCODER_LAYERS,
      num_decoder_layers=NUM_DECODER_LAYERS,
      dim_feedforward=DIM_FEEDFORWARD,
      dropout=DROPOUT,
      max_seq_len=MAX_LEN,
      pad_id=PAD_ID,
  ):
    super().__init__()
    self.pad_id = pad_id
    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
    self.encoder_layers = nn.ModuleList([
        RNNAugmentedEncoderLayer(
            d_model, nhead, enc_gru_hidden, dim_feedforward, dropout
        )
        for _ in range(num_encoder_layers)
    ])
    self.decoder_layers = nn.ModuleList([
        RNNAugmentedDecoderLayer(
            d_model, nhead, dec_gru_hidden, dim_feedforward, dropout
        )
        for _ in range(num_decoder_layers)
    ])
    self.output_proj = nn.Linear(d_model, vocab_size)
    self.embed_scale = math.sqrt(d_model)

  def make_padding_mask(self, seq):
    return seq == self.pad_id

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
      x = layer(
          x,
          memory,
          tgt_mask=tgt_mask,
          tgt_key_padding_mask=tgt_key_padding_mask,
          memory_key_padding_mask=memory_key_padding_mask,
      )
    return self.output_proj(x)


# ==============================================================================
# 3. Tokenizer และ Checkpoint Loading
# ==============================================================================
print("[*] กำลังโหลด SentencePiece Tokenizer...")
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
vocab_size = sp.get_piece_size()


def load_checkpoint_safe(model_path):
  if not os.path.exists(model_path):
    print(f"[!] ไม่พบไฟล์ {model_path}")
    return None
  model = RNNAugmentedTransformer(vocab_size=vocab_size).to(device)
  ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
  model.load_state_dict(ckpt["model_state_dict"])
  model.eval()
  del ckpt
  gc.collect()
  print(f"[+] โหลดโมเดล {model_path} สำเร็จ")
  return model


model_en_th = load_checkpoint_safe(MODEL_EN_TH_PATH)
model_th_en = load_checkpoint_safe(MODEL_TH_EN_PATH)


# ==============================================================================
# 4. ฟังก์ชัน Translation ตรงตาม Colab V5
# ==============================================================================
def encode_text(text: str, add_bos_eos: bool = True):
  ids = sp.encode(text, out_type=int)
  if add_bos_eos:
    ids = [BOS_ID] + ids + [EOS_ID]
  return ids[:MAX_LEN]


def decode_ids(ids):
  ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
  return sp.decode(ids)


def get_banned_ngram_tokens(ngram_tokens, n):
  if len(ngram_tokens) < n:
    return []
  banned = []
  ngram_prefix = tuple(ngram_tokens[-(n - 1) :])
  for i in range(len(ngram_tokens) - n + 1):
    if tuple(ngram_tokens[i : i + n - 1]) == ngram_prefix:
      banned.append(ngram_tokens[i + n - 1])
  return banned


@torch.no_grad()
def greedy_translate(
    model,
    src_ids,
    max_new_tokens=MAX_NEW_TOKENS,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    repetition_penalty=REPETITION_PENALTY,
):
  model.eval()
  src = torch.tensor([src_ids], dtype=torch.long, device=device)
  memory, memory_key_padding_mask = model.encode(src)

  safe_max_len = min(max_new_tokens, MAX_LEN - 1)
  tgt_ids = [BOS_ID]

  for _ in range(safe_max_len):
    tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    logits = model.decode(
        tgt, memory, memory_key_padding_mask=memory_key_padding_mask
    )
    next_token_logits = logits[0, -1, :].clone()

    for tok in set(tgt_ids):
      if next_token_logits[tok] > 0:
        next_token_logits[tok] /= repetition_penalty
      else:
        next_token_logits[tok] *= repetition_penalty

    banned_tokens = get_banned_ngram_tokens(tgt_ids, no_repeat_ngram_size)
    for tok in banned_tokens:
      next_token_logits[tok] = float("-inf")

    next_token = torch.argmax(next_token_logits).item()
    tgt_ids.append(next_token)

    if next_token == EOS_ID:
      break

  return tgt_ids


def translate_sentence(model, text: str) -> str:
  if not text or not text.strip():
    return ""
  src_ids = encode_text(text.strip())
  out_ids = greedy_translate(model, src_ids)
  return decode_ids(out_ids)


# ==============================================================================
# 5. FastAPI Endpoints
# ==============================================================================
app = FastAPI(title="CS EN-TH Bidirectional Translation API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslationRequest(BaseModel):
  text: str
  direction: str = "en-th"


class TranslationResponse(BaseModel):
  translated_text: str
  direction: str


@app.get("/")
def health_check():
  return {
      "status": "online",
      "models_loaded": {
          "en-th": model_en_th is not None,
          "th-en": model_th_en is not None,
      },
  }


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
  direction = req.direction.strip().lower()

  if direction == "th-en":
    target_model = model_th_en
  else:
    direction = "en-th"
    target_model = model_en_th

  if target_model is None:
    raise HTTPException(
        status_code=503,
        detail=(
            f"โมเดลสำหรับทิศทาง {direction} ยังไม่พร้อมใช้งาน"
            " กรุณาตรวจสอบไฟล์ checkpoint"
        ),
    )

  result = translate_sentence(target_model, req.text)
  return TranslationResponse(translated_text=result, direction=direction)