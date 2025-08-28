from __future__ import annotations
import os, re, textwrap
from typing import Optional, List
from ..registry import tool
from ..state import State
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI
from langchain_google_community import SpeechToTextLoader
from pathlib import Path


# =========================
# OpenAI helpers (robust)
# =========================

# Try to import youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
except Exception:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = NoTranscriptFound = CouldNotRetrieveTranscript = Exception  # harmless aliases


_YT_ID_PATTERNS = [
    r"(?:v=)([A-Za-z0-9_\-]{11})",        # https://www.youtube.com/watch?v=VIDEOID
    r"(?:youtu\.be/)([A-Za-z0-9_\-]{11})",# https://youtu.be/VIDEOID
    r"(?:embed/)([A-Za-z0-9_\-]{11})",    # https://www.youtube.com/embed/VIDEOID
]


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
        "Answer with the final answer only. If it is a one word answer that only provide that, no introductions or tags like FINAL ANSWER: necessary"
        "If unknown, reply 'unknown'. " + (sys_hint or "")
    )
    user_msg = f"Question:\n{question}\n\nTranscript:\n{context}"   
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
    
def _extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    # If the user already gave an 11-char ID, accept it
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", url_or_id):
        return url_or_id
    for pat in _YT_ID_PATTERNS:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return None
    
def _fetch_yt_transcript_text(
    video_id: str,
    lang_prefs: Optional[List[str]] = None,
    allow_translate_to: Optional[str] = "en",
) -> str:
    """
    Fetches transcript segments and concatenates them into plain text.
    Resilient to older/newer youtube-transcript-api versions.
    """
    if YouTubeTranscriptApi is None:
        return "ERROR: youtube-transcript-api not installed. Run: pip install youtube-transcript-api"

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.list(video_id)
    print(transcript)

    fetched_transcript = ytt_api.fetch(video_id, languages=["en-US"])
    parts = []
    for snippet in fetched_transcript:
        print(snippet.text)
        parts.append(snippet.text)
    combined = " ".join(parts).strip() 
    return combined or "ERROR: fetched empty transcript."





# =========================
# TOOLS
# =========================



@tool("asr")
def asr_tool(state: State, file_path: Optional[str] = None, **kwargs) -> str:
    """
    Transcribe an audio file (mp3/wav/m4a) and answer the current question using the transcript.
    Return format: final answer only (per question’s instruction).
    """
    client = OpenAI()  # needs OPENAI_API_KEY in your env
    with open(file_path, "rb") as f:
        # "whisper-1" is the classic; some accounts also have "gpt-4o-transcribe"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    print(transcript.text)
    # 2) Ask LLM to produce the final, formatted answer per the question’s instruction
    question = state.get("question") or kwargs.get("question") or ""

    system = (
        "You are an extraction assistant. You will be given a question with strict formatting rules, "
        "and a transcript of an audio clip. Follow the question's instructions EXACTLY and return ONLY the 1-2 word answer, no need of introduction, commentary or even the tags like 'final answer:'"
    )
    user = (
        f"Question (follow formatting strictly):\n{question}\n\n"
        f"Transcript:\n{transcript.text}\n\n"
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
    
@tool("yt_transcript")
def yt_transcript_tool(
    state: State,
    url: Optional[str] = None,
    video_id: Optional[str] = None,
    question: Optional[str] = None,
    lang_prefs: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """
    Pull a YouTube transcript and answer the current question using the LLM.
    Inputs:
      - url: a full YouTube URL (watch, youtu.be, or embed)   (optional if video_id given)
      - video_id: 11-char YouTube ID                          (optional if url given)
      - question: if omitted, will use state['question'] or return transcript directly
      - lang_prefs: e.g., ["en","en-US"] -> preferred transcript languages
    Output:
      - final answer only (if a question is present), else the raw transcript text.
    """
    # Resolve video id
    vid = _extract_video_id(video_id or url or "")
    print(f"Extracted video ID: {vid}")
    if not vid:
        return "ERROR: Provide a valid YouTube URL or 11-character video_id."

    # Fetch transcript text
    transcript = _fetch_yt_transcript_text(vid, lang_prefs=lang_prefs or ["en","en-US","en-GB"])
    print(f"Fetched transcript length: {len(transcript)} chars")
    print(f"Transcript preview: {transcript[:200]}")
    if transcript.startswith("ERROR:"):
        return transcript

    # If there is no question, be useful and return the transcript as-is
    q = question or state.get("question") or kwargs.get("q") or ""
    if not q:
        return transcript

    # Answer using your robust LLM helper (answers 'final answer only')
    # Add a tiny system hint to keep answers concise and factual
    sys_hint = "Keep answers concise and quote exact figures if present."
    return _llm_answer(question=q, context=transcript, sys_hint=sys_hint)

