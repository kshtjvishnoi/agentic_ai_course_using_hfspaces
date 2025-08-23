from __future__ import annotations
import re, html
from typing import Optional, List
from ..registry import tool
from ..state import State
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI

# --- Search client import (ddgs -> fallback to duckduckgo_search) ---
_HAS_DDGS = False
DDGS = None
try:
    # New package name (recommended): pip install ddgs
    from ddgs import DDGS  # type: ignore
    _HAS_DDGS = True
except Exception:
    try:
        # Legacy package name (deprecated): pip install duckduckgo-search
        from duckduckgo_search import DDGS  # type: ignore
        _HAS_DDGS = True
    except Exception:
        DDGS = None
        _HAS_DDGS = False

# --- Optional HTML parsing ---
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None


# ----- helpers -----

def _openai_client():
    return OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def _first_n_chars(s: str, n: int = 6000) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]

def _fetch_url_text(url: str, timeout: int = 12) -> str:
    """Fetch and extract readable text from a web page."""
    import requests
    if BeautifulSoup is None:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (AgenticGraph/1.0; +https://example.local)"
        }
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()
        text = " ".join(
            el.get_text(" ", strip=True)
            for el in soup.find_all(["p", "li", "h2", "h3"])
        )
        return _first_n_chars(html.unescape(text))
    except Exception:
        return ""

def _llm_answer(question: str, context: str, sys_hint: str = "") -> str:
    client = _openai_client()
    if not client:
        return "OPENAI_API_KEY missing."
    system = (
        "You are a precise research assistant. Answer the question ONLY from the provided context. "
        "If the answer is uncertain from the context, say 'unknown'. " + (sys_hint or "")
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Give the final answer only, no citations, no preamble."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


# ----- tools -----

@tool("web_search")
def web_search_tool(state: State, query: Optional[str] = None, k: int = 5, **kwargs) -> str:
    """
    Search the web (DuckDuckGo via ddgs), fetch top pages, and have the LLM answer the current question
    using ONLY those pages' content.
    """
    if not _HAS_DDGS:
        return "ERROR: web_search requires the 'ddgs' (or legacy 'duckduckgo-search') package."
    if BeautifulSoup is None:
        return "ERROR: web_search requires 'beautifulsoup4' installed."

    q = (query or state["question"]).strip()
    results: List[dict] = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(q, max_results=k):
                if r and (r.get("href") or r.get("url")):
                    results.append(r)
    except Exception as e:
        return f"ERROR: search failed: {e}"

    if not results:
        return "ERROR: no search results."

    parts = []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        txt = _fetch_url_text(url)
        if txt:
            title = r.get("title") or ""
            parts.append(f"[{title}] {txt}")

    context = _first_n_chars("\n\n---\n\n".join(parts), 8000)
    if not context:
        return "ERROR: could not extract text from results."

    return _llm_answer(state["question"], context)


@tool("wiki_lookup")
def wiki_lookup_tool(state: State, title_or_query: Optional[str] = None, **kwargs) -> str:
    """
    Pull content from English Wikipedia and answer the current question from it.
    Tries LlamaIndex WikipediaReader if available; otherwise fetches the /wiki/<Title> page directly.
    """
    q = (title_or_query or state["question"]).strip()

    # Optional: LlamaIndex WikipediaReader
    try:
        from llama_index.readers.wikipedia import WikipediaReader  # type: ignore
        reader = WikipediaReader()
        # If short query, treat as page title; else try search mode (some versions support 'search')
        docs = None
        try:
            if len(q.split()) <= 6:
                docs = reader.load_data(pages=[q])
            else:
                # some versions support search=..., others only pages=[...]
                docs = reader.load_data(search=q)  # type: ignore
        except Exception:
            # Fallback: try as pages
            docs = reader.load_data(pages=[q])
        text = _first_n_chars("\n\n".join(getattr(d, "text", "") for d in docs or []), 8000)
        if text:
            return _llm_answer(state["question"], text)
    except Exception:
        pass  # fall through to direct fetch

    # Direct fetch fallback
    slug = q.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{slug}"
    page_text = _fetch_url_text(url)
    if page_text:
        return _llm_answer(state["question"], page_text, sys_hint="If you don't find it here, reply 'unknown'.")
    return "ERROR: wiki content not found."
