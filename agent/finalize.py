# agents/finalize.py
import re
from typing import Dict, Any
from .state import State

_NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,
}

def _words_to_int_simple(text: str) -> int | None:
    t = (text or "").lower().strip().replace("–","-").replace("—","-")
    parts = [p for p in re.split(r"[\s-]+", t) if p]
    total = 0
    for p in parts:
        if p not in _NUM_WORDS:
            return None
        total += _NUM_WORDS[p]
    return total

def _numeric_only(s: str) -> str | None:
    s = (s or "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        v = m.group(0)
        return str(int(v)) if re.fullmatch(r"-?\d+", v) else v
    w = _words_to_int_simple(s)
    return str(w) if w is not None else None

def _first_name_only(s: str) -> str:
    m = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", s or "")
    return m.group(0) if m else (s or "").strip()

def _ioc_code_only(s: str) -> str | None:
    m = re.search(r"\b[A-Z]{3}\b", (s or "").upper())
    return m.group(0) if m else None

def _csv_normalize(items_text: str, alphabetize: bool) -> str:
    raw = re.split(r"[,\n;]+", items_text or "")
    items = [x.strip() for x in raw if x.strip()]
    if alphabetize:
        items = sorted(items, key=str.casefold)
    return ", ".join(items)

def _usd_two_decimals(s: str) -> str | None:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    if not m:
        return None
    try:
        return f"${float(m.group(0)):.2f}"
    except Exception:
        return None

def _san_only(s: str) -> str | None:
    m = re.search(r"\b(O-O(-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?)\b", s or "")
    return m.group(0) if m else None

def normalize_answer(question: str, text: str) -> str:
    q = (question or "").lower()
    a = (text or "").strip()

    if any(kw in q for kw in ["how many", "highest number", "final numeric output", "number of", "least number"]):
        n = _numeric_only(a)
        if n is not None:
            return n

    if "give only the first name" in q:
        return _first_name_only(a)

    if "ioc country code" in q:
        code = _ioc_code_only(a)
        if code:
            return code

    if any(kw in q for kw in ["comma separated", "comma-separated", "comma delimited", "comma-delimited", "ascending order", "alphabetize"]):
        alpha = ("alphabetize" in q) or ("ascending order" in q)
        return _csv_normalize(a, alphabetize=alpha)

    if "usd" in q and ("two decimal" in q or "two decimals" in q):
        usd = _usd_two_decimals(a)
        if usd:
            return usd

    if "algebraic notation" in q or "san" in q:
        san = _san_only(a)
        if san:
            return san

    return a

def is_plausible_for_question(question: str, normalized: str) -> bool:
    q = (question or "").lower()
    a = (normalized or "").strip()

    if any(kw in q for kw in ["how many", "highest number", "final numeric output", "least number", "number of"]):
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", a))
    if "ioc country code" in q:
        return bool(re.fullmatch(r"[A-Z]{3}", a))
    if "algebraic notation" in q or "san" in q:
        return bool(_san_only(a))
    if any(kw in q for kw in ["comma separated", "comma-separated", "comma delimited", "comma-delimited"]):
        return ("," in a)
    if "usd" in q and ("two decimal" in q or "two decimals" in q):
        return bool(re.fullmatch(r"\$\d+\.\d{2}", a))
    if "give only the first name" in q:
        return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", a))
    return len(a) > 0

def early_finish(question: str, observation: str, current_answer: str = "") -> tuple[bool, str]:
    candidate = (observation or "").strip() or (current_answer or "")
    if not candidate:
        return (False, "")
    final = normalize_answer(question, candidate)
    return (is_plausible_for_question(question, final), final)

def finalize_node(state: State) -> Dict[str, Any]:
    raw = state.get("next_action", {}).get("finish") or state.get("answer", "")
    question = state.get("question", "")
    final = normalize_answer(question, str(raw))
    return {"answer": final}
