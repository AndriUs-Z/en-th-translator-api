"""
retrain.py - Fine-tune โมเดลแปล (RNN-augmented Transformer + SentencePiece) ตาม Cell 10 ของ Notebook
ข้อมูลใหม่ = feedback คำแปลที่ user แก้ และ admin กด approve แล้ว (ตาราง word/translation/status/direction)
ทำทีละทิศทาง:  python retrain/retrain.py --direction <th-en|en-th>   (รันจาก root ของ repo)

ขั้นตอน (เหมือน Cell 10 ฉบับ TH-EN):
 1. โหลดโมเดลฐาน (best_model*.pt) ทุกครั้ง  -> เทรนใหม่จากฐานด้วยคำศัพท์ approved ทั้งหมด (ไม่เทรนซ้อนทับของเดิม)
 2. ผสมข้อมูล: general replay + technical replay + glossary (oversample)
 3. freeze encoder/embedding, เรียนเฉพาะ decoder + output_proj
 4. loss = CE + KD (KD เฉพาะตัวอย่าง replay)  + Forgetting Monitor
 5. Weight interpolation เลือก alpha ที่มากที่สุดที่ general val loss ไม่แย่เกินเกณฑ์
 6. บันทึก checkpoint (ถ้า alpha > 0)

ข้อมูล replay/val ต้อง export จาก Notebook ก่อน (tools/export_replay_data.py) แล้วใส่ใน data/
"""
import argparse
import copy
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from supabase import create_client
from torch.utils.data import DataLoader, Dataset

from model_def import BOS_ID, EOS_ID, MAX_LEN, PAD_ID, RNNAugmentedTransformer

# ---------------- Paths / Env ----------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
JOBS = json.loads(os.getenv("JOBS_JSON", "{}") or "{}") or {}   # {"th-en": "<job_id>", ...}

TARGET_DIR = os.getenv("TARGET_DIR", ".")             # root ของ repo API นี้ (มี sp_en_th.model และไฟล์ finetuned)
BASE_DIR = os.getenv("BASE_DIR", "retrain/base")      # โมเดลฐาน best_model*.pt (ดาวน์โหลดจาก Release ตอนรัน)
DATA_DIR = os.getenv("DATA_DIR", "retrain/data")      # replay/val jsonl
TABLE_PAIRS = os.getenv("TABLE_PAIRS", "your_table_name")  # TODO: ชื่อตารางจริง (columns: word, translation, status, direction)

# ---------------- Hyperparameters ----------------
# ค่าต่อทิศทางมาจาก Cell 10 ของแต่ละ Notebook (EN-TH เดิมไม่มี KD จึงตั้ง kd_weight=0)
DIRECTIONS = {
    "th-en": dict(base_ckpt="best_model_th_en.pt", out="finetuned_model_th_en.pt",
                  replay="replay_th_en.jsonl", val="general_val_th_en.jsonl",
                  lr=1e-5, batch=16, accum=2, kd_weight=1.0),
    "en-th": dict(base_ckpt="best_model.pt", out="finetuned_model.pt",
                  replay="replay_en_th.jsonl", val="general_val_en_th.jsonl",
                  lr=1.5e-5, batch=8, accum=2, kd_weight=0.0),
}
SP_FILENAME = "sp_en_th.model"   # อยู่ที่ root ของ repo API

EPOCHS = int(os.getenv("EPOCHS", "2"))
CLIP_NORM = 0.5
LABEL_SMOOTHING = 0.05
KD_TEMP = 2.0
CS_MAX_REPEAT = 5
CS_DEV_FRACTION = float(os.getenv("CS_DEV_FRACTION", "0"))  # 0 = เทรนด้วย feedback ทั้งหมด (แนะนำ); >0 = กันบางส่วนไว้วัดผล (ใช้เมื่อมี >=50 คู่)
FORGET_TOL = 0.02                       # general val loss แย่ลงได้ไม่เกิน 2%
ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.2]
SEED = int(os.getenv("SEED", "42"))
MIN_PAIRS = int(os.getenv("MIN_PAIRS", "100"))   # ต้องมี feedback approved อย่างน้อยกี่คู่ถึงจะเทรน

# ขนาดชุดข้อมูล: ค่า default ลดลงจาก Notebook (25k/10k/2k) เพราะรันบน CPU ของ GitHub
# เพิ่มได้ผ่าน env/Variables ถ้ายอมให้ใช้เวลานานขึ้น (ดูหมายเหตุในคู่มือ)
N_GENERAL_REPLAY = int(os.getenv("N_GENERAL_REPLAY", "6000"))
N_TECHNICAL_REPLAY = int(os.getenv("N_TECHNICAL_REPLAY", "2000"))
N_CS_OVERSAMPLE = int(os.getenv("N_CS_OVERSAMPLE", "1000"))
N_GENERAL_VAL = int(os.getenv("N_GENERAL_VAL", "1000"))

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 2)


def now():
    return datetime.now(timezone.utc).isoformat()


def update_job(direction, **fields):
    job_id = JOBS.get(direction)
    if job_id:
        sb.table("retrain_jobs").update(fields).eq("id", job_id).execute()


# ---------------- Data ----------------
def norm_direction(v):
    return (v or "").strip().lower().replace("_", "-").replace(" ", "")


def has_thai(text):
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


def fetch_cs_pairs(direction):
    """Feedback ที่ user แก้และ admin approve แล้ว ของทิศทางนี้ -> [(word, translation)]  (word = ต้นทาง)"""
    rows, start, page = [], 0, 1000
    while True:
        res = (sb.table(TABLE_PAIRS).select("word,translation,direction")
               .eq("status", "approved").range(start, start + page - 1).execute())
        rows.extend(res.data)
        if len(res.data) < page:
            break
        start += page

    pairs, seen, bad_lang = [], set(), 0
    by_src = {}
    for r in rows:
        if norm_direction(r.get("direction")) != direction:
            continue
        w, t = (r.get("word") or "").strip(), (r.get("translation") or "").strip()
        if not w or not t:
            continue
        # sanity check: th-en ต้นทางต้องมีอักษรไทย / en-th ปลายทางต้องมีอักษรไทย (กัน direction ติดผิด)
        if (direction == "th-en" and not has_thai(w)) or (direction == "en-th" and not has_thai(t)):
            bad_lang += 1
            continue
        by_src.setdefault(w.lower(), set()).add(t)
        if (w, t) in seen:      # ตัดคู่ซ้ำทุกตัวอักษร
            continue
        seen.add((w, t))
        pairs.append((w, t))
    conflicts = sum(1 for v in by_src.values() if len(v) > 1)
    print(f"[{direction}] feedback approved={len(pairs)} | ข้ามเพราะภาษาไม่ตรงทิศทาง={bad_lang} "
          f"| คำต้นทางที่มีคำแปลขัดกัน={conflicts}")
    return pairs


def read_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบ {path} - ต้อง export จาก Notebook แล้วใส่ใน data/ ก่อน")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PairDataset(Dataset):
    """pairs = [(src_text, tgt_text)], flags = [is_glossary]  (encode เหมือน encode_text ใน API)"""
    def __init__(self, pairs, sp, flags=None):
        self.pairs, self.sp = pairs, sp
        self.flags = flags if flags is not None else [False] * len(pairs)

    def _enc(self, text):
        ids = [BOS_ID] + self.sp.encode(text, out_type=int) + [EOS_ID]
        return ids[:MAX_LEN]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, t = self.pairs[i]
        return self._enc(s), self._enc(t), self.flags[i]


def collate(batch):
    def pad(seqs):
        m = max(len(x) for x in seqs)
        return torch.tensor([x + [PAD_ID] * (m - len(x)) for x in seqs], dtype=torch.long)
    return pad([b[0] for b in batch]), pad([b[1] for b in batch]), torch.tensor([bool(b[2]) for b in batch])


def make_loader(pairs, sp, bs, shuffle=False, flags=None, drop_last=False):
    return DataLoader(PairDataset(pairs, sp, flags), batch_size=bs, shuffle=shuffle,
                      collate_fn=collate, drop_last=drop_last)


# ---------------- Loss ----------------
@torch.no_grad()
def eval_loss(model, loader):
    """CE เฉลี่ยต่อ token (ไม่รวม PAD)"""
    if loader is None:
        return float("nan")
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    tot, n = 0.0, 0
    for src, tgt, _ in loader:
        dec_in, label = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, dec_in)
        tot += crit(logits.reshape(-1, logits.size(-1)), label.reshape(-1)).item()
        n += (label != PAD_ID).sum().item()
    return tot / max(n, 1)


def kd_loss_fn(student_logits, teacher_logits, labels, keep_mask, T):
    s_logp = F.log_softmax(student_logits.float() / T, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / T, dim=-1)
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(-1)
    valid = (labels != PAD_ID) & keep_mask.unsqueeze(1)
    return (kl * valid).sum() / valid.sum().clamp(min=1) * (T * T)


# ---------------- Main training ----------------
def train_direction(direction):
    cfg = DIRECTIONS[direction]
    rng = random.Random(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    update_job(direction, status="running", started_at=now())

    # 1. glossary จาก Supabase
    cs_pairs = fetch_cs_pairs(direction)
    print(f"[{direction}] approved glossary pairs: {len(cs_pairs)}")
    if len(cs_pairs) < MIN_PAIRS:
        raise RuntimeError(f"ข้อมูล approved ไม่พอ ({len(cs_pairs)} < {MIN_PAIRS})")

    # 2. tokenizer + โมเดลฐาน
    sp = spm.SentencePieceProcessor(model_file=os.path.join(TARGET_DIR, SP_FILENAME))
    model = RNNAugmentedTransformer(vocab_size=sp.get_piece_size()).to(device)
    base_path = os.path.join(BASE_DIR, cfg["base_ckpt"])
    ckpt = torch.load(base_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    use_kd = cfg["kd_weight"] > 0
    teacher = None
    if use_kd:
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
    print(f"[{direction}] loaded base {base_path} (val_loss เดิม={ckpt.get('val_loss')})")
    del ckpt

    # 3. เตรียมข้อมูล
    cs_shuf = cs_pairs[:]
    rng.shuffle(cs_shuf)
    n_dev = max(1, int(len(cs_shuf) * CS_DEV_FRACTION)) if (CS_DEV_FRACTION > 0 and len(cs_shuf) >= 50) else 0
    cs_train = cs_shuf[n_dev:]
    if n_dev:
        cs_dev = cs_shuf[:n_dev]          # holdout จริง (ไม่ถูกเทรน)
    else:
        cs_dev = cs_train[:200]           # ไม่กันออก: "cs dev" = วัดว่าโมเดลเรียนคำแก้เหล่านี้ได้แค่ไหน (fit ไม่ใช่ generalization)

    replay = read_jsonl(os.path.join(DATA_DIR, cfg["replay"]))
    general = [(r["src"], r["tgt"]) for r in replay if r.get("kind") == "general"]
    technical = [(r["src"], r["tgt"]) for r in replay if r.get("kind") == "technical"]
    general = rng.sample(general, min(N_GENERAL_REPLAY, len(general)))
    technical = rng.sample(technical, min(N_TECHNICAL_REPLAY, len(technical)))
    val_all = [(r["src"], r["tgt"]) for r in read_jsonl(os.path.join(DATA_DIR, cfg["val"]))]
    general_val = rng.sample(val_all, min(N_GENERAL_VAL, len(val_all)))

    repeat = min(CS_MAX_REPEAT, (N_CS_OVERSAMPLE // max(len(cs_train), 1)) + 1)
    cs_over = (cs_train * repeat)[:N_CS_OVERSAMPLE]
    mixed = [(p, False) for p in general] + [(p, False) for p in technical] + [(p, True) for p in cs_over]
    random.shuffle(mixed)
    print(f"[{direction}] mixed: general={len(general):,} technical={len(technical):,} "
          f"cs={len(cs_over):,} (unique {len(cs_train)}, x{repeat}) | cs_dev={len(cs_dev)} | general_val={len(general_val)}")

    val_loader = make_loader(general_val, sp, cfg["batch"])
    dev_loader = make_loader(cs_dev, sp, cfg["batch"]) if cs_dev else None

    def eval_losses():
        return eval_loss(model, val_loader), eval_loss(model, dev_loader)

    # 4. baseline
    base_gen, base_cs = eval_losses()
    gen_limit = base_gen * (1 + FORGET_TOL)
    print(f"[{direction}] baseline general={base_gen:.4f} (limit {gen_limit:.4f}) | cs dev={base_cs:.4f}")

    # 5. freeze encoder/embedding/pos_encoding
    for m in (model.encoder_layers, model.embedding, model.pos_encoding):
        for p in m.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)
    loader = make_loader([m[0] for m in mixed], sp, cfg["batch"], shuffle=True,
                         flags=[m[1] for m in mixed], drop_last=True)

    # 6. fine-tune loop (CE + KD) + forgetting monitor
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        tot_ce = tot_kd = 0.0
        n_ok = 0
        for step, (src, tgt, is_cs) in enumerate(loader):
            dec_in, label = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, dec_in)
            ce = crit(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
            if use_kd:
                with torch.no_grad():
                    t_logits = teacher(src, dec_in)
                kd = kd_loss_fn(logits, t_logits, label, ~is_cs, KD_TEMP)
            else:
                kd = torch.zeros(())
            loss = (ce + cfg["kd_weight"] * kd) / cfg["accum"]
            loss.backward()
            if (step + 1) % cfg["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=CLIP_NORM)
                opt.step()
                opt.zero_grad()
            tot_ce += ce.item(); tot_kd += kd.item(); n_ok += 1
            if step % 50 == 0:
                print(f"[{direction}] epoch {epoch}/{EPOCHS} step {step}/{len(loader)} ce={ce.item():.4f}", flush=True)

        gen_loss, cs_loss = eval_losses()
        print(f"[{direction}] epoch {epoch} | CE={tot_ce/max(n_ok,1):.4f} KD={tot_kd/max(n_ok,1):.4f} "
              f"| general={gen_loss:.4f} (base {base_gen:.4f}) | cs dev={cs_loss:.4f} (base {base_cs:.4f})")
        if gen_loss > gen_limit:
            print(f"[{direction}] Forgetting monitor: general val เกินขีดจำกัด -> หยุด (จะ interpolate กลับ)")
            break

    # 7. weight interpolation: เลือก alpha มากที่สุดที่ไม่ลืมเกินเกณฑ์
    ft_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def load_interpolated(alpha):
        new = {k: (((1 - alpha) * v + alpha * ft_state[k]) if v.dtype.is_floating_point else ft_state[k])
               for k, v in pre_state.items()}
        model.load_state_dict(new)

    chosen_alpha, chosen_gen, chosen_cs = 0.0, base_gen, base_cs
    for alpha in ALPHAS:
        load_interpolated(alpha)
        g, c = eval_losses()
        ok = g <= gen_limit
        print(f"[{direction}] alpha={alpha:.1f} | general={g:.4f} | cs dev={c:.4f} | {'ผ่าน' if ok else 'ลืมเกิน'}")
        if ok:
            chosen_alpha, chosen_gen, chosen_cs = alpha, g, c
            break

    summary = (f"alpha={chosen_alpha:.1f} | general {base_gen:.3f}->{chosen_gen:.3f} "
               f"| cs dev {base_cs:.3f}->{chosen_cs:.3f} | glossary {len(cs_pairs)} คู่")
    print(f"[{direction}] >>> {summary}")

    out_path = os.path.join(TARGET_DIR, cfg["out"])
    if chosen_alpha == 0.0:
        # ไม่มี alpha ผ่านเกณฑ์ -> ไม่ทับโมเดลที่ใช้งานอยู่ (ต่างจาก Cell 10 เดิมที่เซฟโมเดลฐานทับ)
        update_job(direction, status="success", finished_at=now(), num_pairs=len(cs_pairs),
                   final_loss=chosen_gen, message="ไม่มี alpha ผ่านเกณฑ์ ไม่เปลี่ยนโมเดลที่ใช้งานอยู่ | " + summary)
        return

    load_interpolated(chosen_alpha)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": sp.get_piece_size(),
        "alpha": chosen_alpha,
        "base_general_val_loss": base_gen,
        "final_general_val_loss": chosen_gen,
    }, out_path)
    print(f"[{direction}] saved -> {out_path}")
    update_job(direction, status="success", finished_at=now(), num_pairs=len(cs_pairs),
               final_loss=chosen_gen, message=summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=list(DIRECTIONS))
    args = ap.parse_args()
    try:
        train_direction(args.direction)
    except Exception as e:
        traceback.print_exc()
        update_job(args.direction, status="failed", finished_at=now(), message=str(e)[:500])
        sys.exit(1)
