import os
import re
import math
import torch
import torch.nn as nn
import sentencepiece as spm
import ahocorasick
from supabase import create_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# จำกัด PyTorch ให้ใช้ 1 Thread ป้องกัน OOM และ CPU contention บน Render
torch.set_num_threads(1)

app = FastAPI(title="Fast CS Translation API")

# ---------------------------------------------------------------------------
# 1. โครงสร้างสถาปัตยกรรมโมเดล (RNN-Augmented Transformer)
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
# 2. เตรียม Tokenizer และ Weights
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPM_PATH = "sp_en_th.model"
MODEL_PATH = "best_model.pt"  # ใช้ best_model.pt เพื่อคงไวยากรณ์หลักที่แม่นยำ

if not os.path.exists(SPM_PATH):
    raise FileNotFoundError(f"Missing {SPM_PATH}")

sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
model = RNNAugmentedTransformer(vocab_size=sp.get_piece_size()).to(device)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] โหลด Weights สำเร็จจาก {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Missing {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 3. โหลดคลังคำศัพท์เฉพาะทางจาก Supabase ด้วย Aho-Corasick Automaton
# ---------------------------------------------------------------------------
AHO_TREE = ahocorasick.Automaton()
GLOSSARY_LOOKUP = {}

def init_supabase_glossary():
    # ดึงค่า URL และ Key จาก Environment (หรือใส่ค่าเริ่มต้นไว้เป็น fallback)
    sb_url = os.environ.get("SUPABASE_URL", "https://hefytemsjkescpihahsg.supabase.co")
    sb_key = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhlZnl0ZW1zamtlc2NwaWhhaHNnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc5NDM5MTQsImV4cCI6MjA5MzUxOTkxNH0.K7rq1n6eodsRT9nxw0YOjLDBIB0WcoghiB3FqlYodbg")

    try:
        supabase = create_client(sb_url, sb_key)
        res = supabase.table("cs_dictionary").select("word, description_th").execute()
        data = res.data if hasattr(res, "data") else []

        count = 0
        for row in data:
            term = row.get("word", "").strip().lower()
            explanation = row.get("description_th", "").strip()
            if term and explanation:
                # คำอธิบายยาวมักมีวงเล็บหรือคำขยาย ให้เลือกเฉพาะคำแปลหลักด้านหน้า
                clean_th = re.split(r'[,( หมายถึง คือ]', explanation)[0].strip()
                target_meaning = clean_th if clean_th else explanation
                
                AHO_TREE.add_word(term, (term, target_meaning))
                GLOSSARY_LOOKUP[term] = target_meaning
                count += 1

        if count > 0:
            AHO_TREE.make_automaton()
            print(f"[Glossary] โหลดคำศัพท์เฉพาะทางจาก Supabase สำเร็จ: {count} คำ")
        else:
            print("[Glossary] ไม่พบข้อมูลคำศัพท์ในฐานข้อมูล")
    except Exception as e:
        print(f"[Glossary Warning] ไม่สามารถเชื่อมต่อ Supabase ได้: {e}")

init_supabase_glossary()

# ---------------------------------------------------------------------------
# 4. Greedy Decode พร้อมระบบป้องกันคำซ้ำ
# ---------------------------------------------------------------------------
def encode_text(text: str):
    ids = sp.encode(text, out_type=int)
    return ([BOS_ID] + ids + [EOS_ID])[:MAX_LEN]

def decode_ids(ids):
    clean_ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(clean_ids)

@torch.inference_mode()
def greedy_translate(
    src_ids: list[int],
    max_new_tokens: int = 100,
    repetition_penalty: float = 1.35,
    no_repeat_ngram_size: int = 3,
):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory, memory_key_padding_mask = model.encode(src)

    dynamic_limit = min(int(len(src_ids) * 2.5) + 10, max_new_tokens, MAX_LEN - 1)
    tgt_ids = [BOS_ID]
    generated_tokens = []

    for _ in range(dynamic_limit):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        next_token_logits = logits[0, -1, :].clone()

        # Multiplicative Repetition Penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            token_counts = torch.bincount(
                torch.tensor(generated_tokens, dtype=torch.long),
                minlength=next_token_logits.size(0)
            ).to(device)
            penalty_mask = token_counts > 0
            next_token_logits = torch.where(
                penalty_mask & (next_token_logits > 0),
                next_token_logits / (repetition_penalty ** token_counts.float()),
                next_token_logits
            )
            next_token_logits = torch.where(
                penalty_mask & (next_token_logits <= 0),
                next_token_logits * (repetition_penalty ** token_counts.float()),
                next_token_logits
            )

        # Strict N-gram Blocking (ห้ามซ้ำกลุ่มคำ 3 token)
        if no_repeat_ngram_size > 0 and len(tgt_ids) >= no_repeat_ngram_size:
            ngram_prefix = tuple(tgt_ids[-(no_repeat_ngram_size - 1):])
            banned_tokens = set()
            for i in range(len(tgt_ids) - no_repeat_ngram_size + 1):
                prev_ngram = tuple(tgt_ids[i : i + no_repeat_ngram_size - 1])
                if prev_ngram == ngram_prefix:
                    banned_tokens.add(tgt_ids[i + no_repeat_ngram_size - 1])
            for b_tok in banned_tokens:
                next_token_logits[b_tok] = -float("inf")

        next_token_logits[UNK_ID] = -float("inf")
        next_token_logits[PAD_ID] = -float("inf")

        next_token = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token)
        generated_tokens.append(next_token)

        if next_token == EOS_ID:
            break

    return decode_ids(tgt_ids)

# ---------------------------------------------------------------------------
# 5. Core Translation Pipeline ผสาน Database อัตโนมัติ (ฉบับแก้ปัญหา)
# ---------------------------------------------------------------------------
def translate_pipeline(text: str) -> str:
  text_lower = text.lower()
  matched_terms = []

  # 1. ตรวจหาคำศัพท์เฉพาะทางจาก Aho-Corasick Automaton
  if len(GLOSSARY_LOOKUP) > 0:
    for end_index, (term, th_val) in AHO_TREE.iter(text_lower):
      start_index = end_index - len(term) + 1
      is_start_ok = (start_index == 0) or not text_lower[
          start_index - 1
      ].isalnum()
      is_end_ok = (end_index == len(text_lower) - 1) or not text_lower[
          end_index + 1
      ].isalnum()
      if is_start_ok and is_end_ok:
        matched_terms.append((start_index, end_index, term, th_val))

  # เรียงลำดับคำยาวขึ้นก่อน ป้องกันการชนกันของคำประสม
  matched_terms.sort(key=lambda x: (x[1] - x[0]), reverse=True)

  # กรองช่วงคำที่ทับซ้อนกันออก
  filtered_matches = []
  occupied = set()
  for start, end, term, th_val in matched_terms:
    span = set(range(start, end + 1))
    if not span.intersection(occupied):
      filtered_matches.append((start, end, term, th_val))
      occupied.update(span)

  # เรียงตามลำดับตำแหน่งในประโยคจากหลังมาหน้า เพื่อแทนที่ Placeholder โดยไม่กระทบ Index
  filtered_matches.sort(key=lambda x: x[0], reverse=True)

  placeholder_map = {}
  masked_text = text
  for i, (start, end, term, th_val) in enumerate(filtered_matches):
    placeholder = f"XTERM{i}X"
    placeholder_map[placeholder] = th_val
    masked_text = masked_text[:start] + placeholder + masked_text[end + 1 :]

  # 2. ส่งประโยคที่ Mask แล้วเข้าโมเดล
  src_ids = encode_text(masked_text)
  translated = greedy_translate(src_ids)

  # 3. นำคำแปลจาก Database ใส่กลับคืนแทน Placeholder
  final_translated = translated
  for placeholder, th_meaning in placeholder_map.items():
    # ใช้ Regex ค้นหา placeholder แบบยืดหยุ่น (เผื่อโมเดลเว้นวรรคหรือเปลี่ยนเป็นตัวพิมพ์เล็ก)
    pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
    final_translated = pattern.sub(th_meaning, final_translated)

  return final_translated
# ---------------------------------------------------------------------------
# 6. API Endpoint
# ---------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(req: TranslationRequest):
    cleaned_input = req.text.strip()
    if not cleaned_input:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = translate_pipeline(cleaned_input)
    return TranslationResponse(translated_text=result)
