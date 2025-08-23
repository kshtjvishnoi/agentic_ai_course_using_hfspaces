# agent/tools/openai_engine.py
import os, time, base64, mimetypes
from typing import Optional
from openai import OpenAI
from ..state import State
from ..finalize import normalize_answer
from ..registry import tool  # your @tool decorator



_CLIENT = None
def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _CLIENT

def _image_part(path: str) -> Optional[dict]:
    if not path:
        return None
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith(("image/", "application/pdf")):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{b64}",
    }

_SYSTEM = (
    "You are a precise research assistant. "
    "Return ONLY the answer, not commentary."
    "Follow the user's instructions exactly. "
    "If the sentence is reversed, put it in a normal order and treat that as your instruction. And simply return back this answer"
    "If the question requests a numeric/count result, output only the number. "
    "If CSV is requested, output a comma-separated list (no bullets, no prose). "
    "If IOC country code is requested, output just the 3-letter code. "
    "If first name only is requested, output just the first name. "
    "If chess algebraic notation is requested, output SAN only (e.g., Qh2#). "
    "If USD two-decimals is requested, format like $123.45. "
    "Do not add explanations unless explicitly asked."
    "For example - Query - Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?"
    "Answer - Dinosaur"
)

@tool("openai_answer")
def openai_answer_tool(
    state: State,
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """
    General-purpose OpenAI LLM tool for tasks without a specific tool.
    - Handles text-only questions.
    - If an image/PDF file is attached (state.file_name), sends it to a vision-capable model.
    - Honors output-shape constraints (numeric only, CSV, IOC code, SAN, USD 2dp, first name).
    Parameters:
      instruction: optional extra guidance (e.g., "answer with a single integer")
      model: override model (default from OPENAI_MODEL or gpt-4o-mini)
    Returns: final text (already normalized for the question).
    """
    q = state.get("question", "")
    fpath = (state.get("file_name") or "").strip()
    model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    parts = [{"type": "text", "text": (instruction + "\n\n" if instruction else "") + f"Question:\n{q}"}]
    img = _image_part(fpath)
    if img:
        parts.append(img)

    # dprint(f"[openai_answer] model={model} temp={temperature} file={'Y' if img else 'N'}")
    t0 = time.perf_counter()
    try:
        rsp = _client().chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": parts},
            ],
        )
        text = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: openai_answer failed: {e}"
    dt = int((time.perf_counter() - t0) * 1000)
    # dprint(f"[openai_answer] {dt}ms → {_short(text, 160)}")

    # normalize to the requested shape for the competition
    return normalize_answer(q, text)
