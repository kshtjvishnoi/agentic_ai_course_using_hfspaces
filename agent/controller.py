import json, re
from typing import Dict, Any
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL
from .state import State
from .registry import tool_allowed, TOOL_REGISTRY
from agent.finalize import early_finish, normalize_answer

SYSTEM = """
You are a tool-using controller. Choose the next action.

Guidance:
- Consider the given plan step (sub-goal + tool hint), but you may override it if a better tool exists. 
- Prefer calling a TOOL at least once before finishing. If the previous step fails then use another tool. Prefer openai_answer tool over wiki_lookup and web_search.

Output ONLY valid JSON:
{"tool": "<tool_name>", "params": {...}, "why": "<1-2 sentences>"}
OR
{"finish": "<final answer>", "why": "<1-2 sentences>"}
"""

def _tool_catalog() -> str:
    """Build a short catalog from registered tools (name + first docstring line)."""
    lines = []
    for name, fn in sorted(TOOL_REGISTRY.items()):
        desc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        desc = desc[:160]
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def controller_node(state: State) -> Dict[str, Any]:

    #print("[debug] keys in state at controller:", list(state.keys()))

    # ---- Compatibility defaults (handle older states) ----
    plan = state.get("plan") or []
    cursor = int(state.get("plan_cursor", 0) or 0)

    # Finish if we already have a good observation and auto-finish is enabled
    if state.get("scratchpad"):
        last = state["scratchpad"][-1]
        if last.get("success") and state.get("auto_finish_after_tool", False):
            return {"next_action": {"finish": last.get("observation",""), "why": "Successful tool result."}}

    if state.get("step", 0) >= state.get("max_steps", 0):
        return {"next_action": {"finish": "Max steps reached.", "why": "Budget exhausted."}}

    # Pull current plan step (if any)
    ps = plan[cursor] if 0 <= cursor < len(plan) else None
    if state.get("plan") and 0 <= state.get("plan_cursor", 0) < len(state["plan"]):
        ps = state["plan"][state["plan_cursor"]]

    # Build readable scratchpad
    turns = []
    scratchpad = state.get("scratchpad", [])
    for t in scratchpad:
        turns.append(
            f"Thought: {t.get('thought','')}\n"
            f"Action: {t.get('action','')} {t.get('params',{})}\n"
            f"Observation: {t.get('observation','')}"
        )
    ctx = "\n\n".join(turns) or "None"
    print("context", ctx)

    plan_str = f"Current plan step: {ps}" if ps else "No plan step."
    catalog = _tool_catalog()
    msg = (
        f"Task: {state['question']}\n"
        f"File: {state.get('file_name') or 'None'}\n\n"
        f"{plan_str}\n\n"
        f"So far:\n{ctx}\n\n"
        f"Available tools:\n{catalog}\n"
        "Return valid JSON only."
    )

    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if not client:
        return {"next_action": {"finish": "OPENAI_API_KEY missing.", "why": "LLM not configured."}}
    
    

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    print("Client response", raw)
    #print("[debug] keys in state at controller:", list(state.keys()))
    try:
        decision = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        decision = json.loads(m.group(0)) if m else {"finish":"Controller did not return JSON.","why":"Parse error."}
    return {"next_action": decision}
