from __future__ import annotations
import os, re, textwrap
from typing import Optional, List
from ..registry import tool
from ..state import State
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI


# =========================
# OpenAI helpers (robust)
# =========================

def _openai_client() -> Optional[OpenAI]:
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    return OpenAI(api_key=key) if key else None


def _extract_text_from_openai_response(resp) -> str:
    """
    Works for both Chat Completions and Responses API objects.
    """
    # Chat Completions style
    if hasattr(resp, "choices"):
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
    # Responses API convenience attr (newer SDKs)
    if hasattr(resp, "output_text"):
        try:
            return (resp.output_text or "").strip()
        except Exception:
            pass
    # Responses API raw structure fallback
    try:
        out = resp.output[0].content[0].text
        return (out or "").strip()
    except Exception:
        return ""


def _llm_answer(question: str, context: str, sys_hint: str = "") -> str:
    client = _openai_client()
    if not client:
        return "ERROR: OPENAI_API_KEY missing."

    system_msg = (
        "You are a precise assistant. Answer ONLY from the given transcript/context. "
        "If unknown, reply 'unknown'. " + (sys_hint or "")
    )
    user_msg = f"Question:\n{question}\n\nTranscript:\n{context}\n\nAnswer with the final answer only."

    # Try Chat Completions first (works for classic chat models).
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        txt = _extract_text_from_openai_response(resp)
        if txt:
            return txt
    except Exception as e:
        # Fall through to Responses API
        pass

    # Responses API fallback (works for newer families)
    try:
        # We fold roles into a single input for maximum compatibility.
        prompt = f"{system_msg}\n\n{user_msg}"
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        txt = _extract_text_from_openai_response(resp)
        if txt:
            return txt
        return "ERROR: empty model response."
    except Exception as e:
        return f"ERROR: LLM call failed: {e}"


# =========================
# TOOLS
# =========================



@tool("asr")
def asr_tool(state: State, file_path: Optional[str] = None, **kwargs) -> str:
    """
    Transcribe an audio file (mp3/wav/m4a) and answer the current question using the transcript.
    Uses OpenAI Whisper API (whisper-1 by default). Return format: final answer only (per question’s instruction).
    """
    client = _openai_client()
    if not client:
        return "ERROR: OPENAI_API_KEY missing."

    path = file_path or state.get("file_name") or state.get("file_path") or ""
    if not path:
        return "ERROR: no audio file provided. Pass file_path or set state['file_name']/['file_path']."

    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"

    # Basic extension hint (not a hard check)
    if not re.search(r"\.(mp3|wav|m4a|mp4|mpga|mpeg|webm|ogg)$", path, re.IGNORECASE):
        # Let the API try anyway, but warn the user
        pass

    # 1) Transcribe
    try:
        with open(path, "rb") as f:
            # If you have access to the newer transcribe model, replace with: model="gpt-4o-transcribe"
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        transcript = (tr.text or "").strip()
        if not transcript:
            return "ERROR: transcription returned empty text."
    except Exception as e:
        return f"ERROR: transcription failed: {e}"

    # 2) Ask LLM to produce the final, formatted answer per the question’s instruction
    question = state.get("question") or kwargs.get("question") or ""
    if not question:
        # If no question, just return the transcript to be useful
        return transcript

    system = (
        "You are an extraction assistant. You will be given a question with strict formatting rules, "
        "and a transcript of an audio clip. Follow the question's instructions EXACTLY and return ONLY the final answer."
    )
    user = (
        f"Question (follow formatting strictly):\n{question}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return ONLY the final answer, nothing else."
    )

    # Try Chat, then Responses
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        out = _extract_text_from_openai_response(resp)
        if out:
            return out
    except Exception:
        pass

    try:
        prompt = f"{system}\n\n{user}"
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        out = _extract_text_from_openai_response(resp)
        if out:
            return out
        return "ERROR: empty model response."
    except Exception as e:
        # fallback: at least return transcript
        return f"ERROR: LLM extraction failed. Transcript: {transcript[:1000]}"
