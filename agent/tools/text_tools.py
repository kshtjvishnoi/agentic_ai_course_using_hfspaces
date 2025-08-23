from openai import OpenAI
from ..registry import tool
from ..config import OPENAI_API_KEY, OPENAI_MODEL
from ..state import State

@tool("reverse_decode")
def reverse_decode_tool(state: State, text: str | None = None, **kwargs) -> str:
    """Pure LLM: choose between original/reversed; follow instruction; return ONLY final answer."""
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if not client:
        return "OPENAI_API_KEY missing."

    if text is None:
        text = kwargs.get("input") or kwargs.get("question") or state["question"]

    original = text
    reversed_q = original[::-1]

    system_prompt = (
        "You are a precise decoder for short word puzzles.\n"
        "You may get a text and its character-wise reversal. Follow these rules:\n"
        "1) Decide which version is meaningful English (original vs. reversed).\n"
        "2) If the meaningful text contains an instruction (e.g., 'write the opposite of \"X\"'):\n"
        "   - Execute it literally.\n"
        "   - For 'opposite', output the standard English antonym of X.\n"
        "   - DO NOT echo the instruction or the original word.\n"
        "3) Return ONLY the final answer, with no extra words, no punctuation, no quotes.\n"
        "4) Self-check: if asked for an 'opposite/antonym', ensure your answer is NOT identical to the target word."
    )
    user_prompt = f"Original:\n{original}\n\nReversed:\n{reversed_q}\n\nReturn ONLY the final answer."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
    )
    return resp.choices[0].message.content.strip()

