# รันเป็น "Cell ใหม่" ใน Colab หลังเซลล์ที่สร้าง train_pairs / val_pairs / opus_pairs แล้ว
# (ก่อน Cell 10) ใช้ใน Notebook ของทิศทาง en-th เท่านั้น
# จะได้ไฟล์ replay_en_th.jsonl และ general_val_en_th.jsonl ให้เอาไปใส่ retrain/data/ ของ repo นี้
import json, os, random

DIRECTION = "en-th"          # (ตั้งไว้ให้แล้วสำหรับ repo นี้)
SEED = 42
N_GENERAL, N_TECHNICAL, N_VAL = 25_000, 10_000, 2_000   # เท่ากับ Cell 10 (retrain.py จะสุ่มลดตามเวลาที่ยอมได้เอง)

tag = DIRECTION.replace("-", "_")
rng = random.Random(SEED)


def to_text(x):
    """รองรับทั้ง pair ที่เป็นข้อความ และที่เป็น list ของ token id"""
    if isinstance(x, str):
        return x
    return sp.decode([i for i in x if i not in (0, 1, 2, 3)])   # PAD, UNK, BOS, EOS


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows):,} rows, {os.path.getsize(path)/1e6:.1f} MB")


general = rng.sample(train_pairs, min(N_GENERAL, len(train_pairs)))
tech_pool = opus_pairs if "opus_pairs" in globals() and opus_pairs else []
technical = rng.sample(tech_pool, min(N_TECHNICAL, len(tech_pool))) if tech_pool else []
val = rng.sample(val_pairs, min(N_VAL, len(val_pairs)))

dump(f"replay_{tag}.jsonl",
     [{"src": to_text(s), "tgt": to_text(t), "kind": "general"} for s, t in general]
     + [{"src": to_text(s), "tgt": to_text(t), "kind": "technical"} for s, t in technical])
dump(f"general_val_{tag}.jsonl", [{"src": to_text(s), "tgt": to_text(t)} for s, t in val])

try:  # ดาวน์โหลดลงเครื่อง (Colab)
    from google.colab import files
    files.download(f"replay_{tag}.jsonl")
    files.download(f"general_val_{tag}.jsonl")
except ImportError:
    pass
