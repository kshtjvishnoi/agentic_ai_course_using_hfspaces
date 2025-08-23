import json, re
from typing import Dict, Any, List
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL
from .state import State, PlanStep
from .registry import TOOL_REGISTRY
from agent.finalize import early_finish, normalize_answer


SYSTEM = """
You are a planning assistant. Produce a short plan (1-4 steps) to solve the task with the available tools.
Each step should include:
- goal: concise sub-goal
- tool_hint: the most appropriate tool name from the catalog (or "" if no clear tool). Prefer the tool openai_answer over web_search in case of searching and reasoning tasks that does not involve additional files. 
- params_hint: minimal params (if any), as a JSON object. Should contain a detailed "query" for the tool.

Return JSON only:
{"steps":[{"goal":"...","tool_hint":"...","params_hint":{...}}, ...]}
"""

def _tool_catalog() -> str:
    lines = []
    for name, fn in sorted(TOOL_REGISTRY.items()):
        desc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        lines.append(f"- {name}: {desc[:160]}")
    return "\n".join(lines)

def _render_question(state: State) -> str:
    return (
        f"Task:\n{state['question']}\n\n"
        f"Attached file: {state.get('file_name') or 'None'}\n\n"
        f"Available tools:\n{_tool_catalog()}\n"
        "Return minimal JSON with 1-4 steps."
    )

def planner_node(state: State) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        # Safe fallback: a single generic step pointing to the controller
        return {"plan": [], "plan_cursor": 0}

    client = OpenAI(api_key=OPENAI_API_KEY)
    msg = _render_question(state)
    print("planner Message", msg)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":msg}],
    )
    raw = resp.choices[0].message.content.strip()
    print("Planner response:", raw)
    try:
        data = json.loads(raw)
        print(data)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {"steps": []}

    steps_in = data.get("steps") or []
    steps: List[PlanStep] = []
    for s in steps_in[:4]:
        steps.append({
            "goal": str(s.get("goal","")).strip(),
            "tool_hint": str(s.get("tool_hint","")).strip(),
            "params_hint": s.get("params_hint") or {},
        })
    print(f"[planner] produced {len(steps)} steps")
    return {"plan": steps, "plan_cursor": 0}
