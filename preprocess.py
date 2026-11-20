"""
preprocess.py - เตรียมข้อความภาษาอังกฤษก่อนส่งเข้าโมเดล EN->TH

โมเดลแยกตัวพิมพ์เล็ก/ใหญ่ ("Jan" กับ "jan" เป็นคนละ token) และเห็นชื่อเดือน/วันแบบตัวใหญ่เป็นหลัก
เช่น "jan 5, 2026" -> "เจียน 52026", "dec 31" -> "ตัดสินใจ 31"
ไฟล์นี้จึงทำชื่อเดือน/วันให้เป็นรูปมาตรฐาน (ตัวใหญ่ และเขียนเต็ม) ก่อนส่งเข้าโมเดล
คำที่กำกวม (may, march, sun, sat, wed ...) จะแปลงเฉพาะเมื่ออยู่ติดตัวเลขหรืออยู่ในช่วงวัน เช่น "mon-fri"
"""
import re

MONTHS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April", "may": "May",
    "jun": "June", "jul": "July", "aug": "August", "sep": "September", "sept": "September",
    "oct": "October", "nov": "November", "dec": "December",
}
FULL_MONTHS = {v.lower(): v for v in MONTHS.values()}
DAYS = {"mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
        "thur": "Thursday", "thurs": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}
FULL_DAYS = {v.lower(): v for v in DAYS.values()}

_MONTH_WORD = "|".join(sorted(set(MONTHS) | set(FULL_MONTHS), key=len, reverse=True))
_DAY_WORD = "|".join(sorted(set(DAYS) | set(FULL_DAYS), key=len, reverse=True))

# เดือนที่อยู่ติดตัวเลข: "jan 5", "Dec. 31", "5 jan", "31 dec 2026"
_MONTH_BEFORE_NUM = re.compile(rf"\b({_MONTH_WORD})\b\.?(?=\s*\d)", re.I)
_MONTH_AFTER_NUM = re.compile(rf"(?<=\d)(\s+)({_MONTH_WORD})\b\.?", re.I)
# ช่วงวัน: "mon-fri", "Mon – Fri", "sat/sun"
_DAY_RANGE = re.compile(rf"\b({_DAY_WORD})\b\.?(\s*[-–/]\s*)({_DAY_WORD})\b\.?", re.I)
# ชื่อวันเต็มที่ไม่กำกวม: monday ... sunday
_FULL_DAY = re.compile(rf"\b({'|'.join(FULL_DAYS)})\b", re.I)
# วันย่อที่ตามด้วยตัวเลข/เวลา: "wed 3pm", "on Fri. 10:00"
_DAY_BEFORE_NUM = re.compile(rf"\b({'|'.join(sorted(DAYS, key=len, reverse=True))})\b\.?(?=,?\s*\d)", re.I)


# วันย่อหลัง "on": "on wed", "on Fri."
_ON_DAY = re.compile(rf"\b(on\s+)({'|'.join(sorted(DAYS, key=len, reverse=True))})\b", re.I)


def _month(word: str) -> str:
    w = word.lower()
    return MONTHS.get(w) or FULL_MONTHS.get(w, word)


def _day(word: str) -> str:
    w = word.lower()
    return DAYS.get(w) or FULL_DAYS.get(w, word)


def preprocess_en(text: str) -> str:
    t = _DAY_RANGE.sub(lambda m: f"{_day(m.group(1))}{m.group(2)}{_day(m.group(3))}", text)
    t = _MONTH_BEFORE_NUM.sub(lambda m: _month(m.group(1)), t)
    t = _MONTH_AFTER_NUM.sub(lambda m: m.group(1) + _month(m.group(2)), t)
    t = _DAY_BEFORE_NUM.sub(lambda m: _day(m.group(1)), t)
    t = _FULL_DAY.sub(lambda m: _day(m.group(1)), t)
    t = _ON_DAY.sub(lambda m: m.group(1) + _day(m.group(2)), t)
    # ตัวอักษรแรกของข้อความ/บรรทัดเป็นตัวใหญ่ (ไม่ใช้จุดเป็นตัวแบ่ง เพราะจะไปโดนตัวย่ออย่าง "p.m. on")
    t = re.sub(r"(^|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t


# ---------------------------------------------------------------------------
# ใส่อักขระที่ tokenizer ไม่รู้จักคืนหลังแปล
# tokenizer ถูกเทรนด้วย character_coverage=0.9995 อักขระที่พบน้อย (@ # อีโมจิ ★) จึงกลายเป็น <unk>
# และโมเดลแปลออกมาเป็น " ⁇ " ฟังก์ชันนี้จำอักขระเหล่านั้นจากต้นฉบับ แล้วใส่คืนตามลำดับ
# ---------------------------------------------------------------------------
UNK_MARK = "⁇"   # ⁇ ที่ SentencePiece ใช้แทน <unk> ตอน decode


def unknown_spans(text, sp):
    unk = sp.unk_id()
    spans, cur = [], ""
    for ch in text:
        is_unk = (not ch.isspace()) and unk in sp.encode(ch, out_type=int)
        if is_unk:
            cur += ch
        elif cur:
            spans.append(cur); cur = ""
    if cur:
        spans.append(cur)
    return spans


def restore_unknown(translated, source, sp):
    if UNK_MARK not in translated:
        return translated
    spans = iter(unknown_spans(source, sp))
    def put_back(m):
        sym = next(spans, "")
        if not sym:
            return " "
        # เคารพการเว้นวรรครอบอักขระตามต้นฉบับ เช่น "test@example.com" ไม่มีช่องว่างรอบ @
        i = source.find(sym)
        before = " " if i > 0 and source[i - 1].isspace() else ""
        after = " " if i + len(sym) < len(source) and source[i + len(sym)].isspace() else ""
        return before + sym + after
    out = re.sub(r"\s*" + UNK_MARK + r"\s*", put_back, translated)
    return re.sub(r"[ \t]{2,}", " ", out).strip()
