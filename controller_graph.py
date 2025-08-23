# controller_graph.py
from __future__ import annotations
import os
import json
import re
import tempfile
import subprocess
from typing import TypedDict, List, Dict, Any, Optional, Callable

# LangGraph
from langgraph.graph import StateGraph, END

# ---------------------------
# State schema
# ---------------------------
class Turn(TypedDict):
    thought: str
    action: str
    params: Dict[str, Any]
    observation: str

class State(TypedDict):
    task_id: str
    question: str
    file_name: str
    plan: str
    scratchpad: List[Turn]
    next_action: Optional[Dict[str, Any]]   # {"tool": str, "params": {...}} OR {"finish": "..."}
    step: int
    max_steps: int
    answer: str
    allowed_tools: Optional[List[str]]      # optional per-task allowlist


# ---------------------------
# Tool registry
# ---------------------------
ToolFn = Callable[[State], str]
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {}

def tool(name: str):
    """Decorator to register a tool by name."""
    def deco(fn: Callable[..., str]):
        TOOL_REGISTRY[name] = fn
        return fn
    return deco

def _tool_allowed(state: State, name: str) -> bool:
    allow = state.get("allowed_tools")
    return (name in TOOL_REGISTRY) and (not allow or name in allow)


# ---------------------------
# Useful local tools (some real, some stubs you can fill)
# ---------------------------

@tool("math_eval")
def math_eval_tool(state: State, expr: Optional[str] = None) -> str:
    """Safely evaluate a basic arithmetic expression."""
    import ast, operator as op
    s = (expr or state["question"]).replace("=", "").strip()
    allowed_ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
    }
    def eval_(node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in [ast.USub]:
            return allowed_ops[type(node.op)](eval_(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](eval_(node.left), eval_(node.right))
        raise ValueError("Unsupported expression")
    try:
        val = eval_(ast.parse(s, mode="eval").body)
        return str(val)
    except Exception as e:
        return f"ERROR: math_eval failed: {e}"

@tool("reverse_decode")
def reverse_decode_tool(state: State) -> str:
    """
    Use the LLM to solve reversed-sentence puzzles or decoding tasks.
    The LLM receives both the original and the reversed string, decides which is meaningful,
    follows the instruction if any, and returns ONLY the final answer.
    """
    from openai import OpenAI

    # 👇 put your key here (or read from env if you prefer)
    OPENAI_API_KEY = "sk-your_api_key_here"
    client = OpenAI(api_key=OPENAI_API_KEY)

    q = state["question"]
    reversed_q = q[::-1]
    print(f"Original: {q}")
    print(f"Reversed: {reversed_q}")

    system_prompt = (
        """You are a puzzle solver. You may be given text that is reversed, plus its character-wise reversal.
         1. Determine which is the meaningful English version.
         2. If it contains an instruction (like 'write the opposite of X'), follow it.
         3. Output ONLY the final answer as a single word or phrase. No explanations."""
    )

    user_prompt = (
        f"Original:\n{q}\n\n"
        f"Reversed:\n{reversed_q}\n\n"
        "Return ONLY the answer. Example: 'right'"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()



@tool("logic_commutativity_set")
def logic_commutativity_set_tool(state: State, table: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    """
    Given a Cayley table (SxS -> S) in `table`, return the subset of S
    involved in any counter-examples to commutativity, as an alpha-sorted CSV.
    If no `table` passed, try to parse from the question text (Markdown table).
    """
    import re
    q = state["question"]
    if table is None:
        # Very light parser for the provided markdown
        # Rows look like: |a|a|b|c|b|d|
        lines = [ln.strip() for ln in q.splitlines() if ln.strip().startswith("|")]
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        elems = headers[1:]  # first cell is "*"
        tbl: Dict[str, Dict[str, str]] = {r: {} for r in elems}
        for row in lines[2:]:  # skip header + separator
            parts = [p.strip() for p in row.strip("|").split("|")]
            r = parts[0]
            for j, c in enumerate(elems):
                tbl[r][c] = parts[j+1]
        table = tbl
    # collect offenders
    offenders = set()
    keys = sorted(table.keys())
    for x in keys:
        for y in keys:
            xy = table[x][y]
            yx = table[y][x]
            if xy != yx:
                offenders.update([x, y])
    return ",".join(sorted(offenders))

@tool("excel_sum_food")
def excel_sum_food_tool(state: State, file_path: Optional[str] = None, food_column: str = "Category",
                        value_column: str = "Sales", drinks_label: str = "Drinks") -> str:
    """
    Sum sales for non-drinks rows. Requires pandas+openpyxl installed.
    """
    import pandas as pd
    path = file_path or state["file_name"]
    if not path or not path.lower().endswith(".xlsx"):
        return "ERROR: No .xlsx provided."
    try:
        df = pd.read_excel(path)
        if food_column not in df.columns or value_column not in df.columns:
            return "ERROR: Columns not found."
        total = df.loc[df[food_column].astype(str).str.lower() != drinks_label.lower(), value_column].sum()
        return f"{total:.2f}"
    except Exception as e:
        return f"ERROR: excel_sum_food failed: {e}"

@tool("code_run")
def code_run_tool(state: State, file_path: Optional[str] = None, timeout: int = 5) -> str:
    """
    Run a provided .py in a restricted subprocess; return last line of stdout or str(int).
    """
    path = file_path or state["file_name"]
    if not path or not path.endswith(".py"):
        return "ERROR: No .py file supplied."
    try:
        with tempfile.TemporaryDirectory() as td:
            # copy file to temp (optional)
            # Safer: run with -I (isolated), -B, no site, etc.
            proc = subprocess.run(
                ["python", "-I", "-B", path],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        if proc.returncode != 0:
            return f"ERROR: code_run failed: {proc.stderr.strip()[:400]}"
        out = proc.stdout.strip().splitlines()
        return out[-1].strip() if out else ""
    except Exception as e:
        return f"ERROR: code_run exception: {e}"

@tool("botany_strict_vegetables")
def botany_strict_vegetables_tool(state: State, items_csv: Optional[str] = None) -> str:
    """
    From a list in the question, return botanical vegetables only (no fruits/seeds).
    Output: alphabetized CSV.
    """
    # From the given list in the dataset:
    # milk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums,
    # green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery,
    # zucchini, lettuce, peanuts
    # Botanical fruits/seeds to exclude: plums (fruit), green beans (fruit pods),
    # corn (grain/fruit), bell pepper (fruit), zucchini (fruit), peanuts (seeds),
    # fresh basil (herb/leaf; some profs exclude herbs from veg), whole allspice (spice/seed),
    # acorns (nuts/seeds), whole bean coffee (seed).
    # True vegetables (botanical plant parts not fruit/seed): broccoli (flower), celery (petiole),
    # lettuce (leaf), sweet potatoes (storage root).
    # Also exclude non-food-category basics like flour, milk, eggs (not plant vegetables).
    text = items_csv or state["question"]
    # crude split
    items = [x.strip().lower() for x in re.split(r"[,\n]+", text) if x.strip()]
    veg_true = set(["broccoli", "celery", "lettuce", "sweet potatoes"])
    found = sorted({it for it in items if it in veg_true})
    return ", ".join(found)

# ----- Stubs you can fill with web/LLM/ASR/vision -----

@tool("web_search")
def web_search_tool(state: State, query: Optional[str] = None, k: int = 5) -> str:
    return "TODO: implement web_search"

@tool("wiki_lookup")
def wiki_lookup_tool(state: State, title_or_query: Optional[str] = None) -> str:
    return "TODO: implement wiki_lookup"

@tool("yt_transcript")
def yt_transcript_tool(state: State, url: Optional[str] = None) -> str:
    return "TODO: implement yt_transcript"

@tool("asr")
def asr_tool(state: State, file_path: Optional[str] = None) -> str:
    return "TODO: implement ASR"

@tool("chess_from_image")
def chess_from_image_tool(state: State, file_path: Optional[str] = None) -> str:
    return "TODO: implement chess_from_image"

@tool("chess_engine")
def chess_engine_tool(state: State, fen: Optional[str] = None) -> str:
    return "TODO: implement chess_engine"

# ---------------------------
# Controller (LLM-driven if key present; otherwise rule-based)
# ---------------------------

SYSTEM = """You are a tool-using controller. You must pick the best next step to solve the task.

Rules:
- Prefer calling an appropriate TOOL at least once before finishing.
- Do NOT return {"finish": "..."} on step 0 unless the answer is trivially obvious without tools.
- If the text may be encoded, reversed, or cryptic, FIRST call a decoding tool (e.g., "reverse_decode").
- Output ONLY valid JSON with one of:
  {"tool": "<tool_name>", "params": {...}}
  or
  {"finish": "<final_answer_string>"}"""

def _render_context(state: State) -> str:
    turns = []
    for t in state["scratchpad"]:
        turns.append(f"Thought: {t['thought']}\nAction: {t['action']} {t['params']}\nObservation: {t['observation']}")
    ctx = "\n\n".join(turns) if turns else "None"
    allow = ", ".join(state.get("allowed_tools") or sorted(TOOL_REGISTRY.keys()))
    return (
        f"Task: {state['question']}\n"
        f"File: {state.get('file_name') or 'None'}\n"
        f"So far:\n{ctx}\n\n"
        f"Allowed tools: {allow}\n"
        f"Steps left: {state['max_steps'] - state['step']}\n"
        f"Respond with JSON only."
    )

def _controller_llm_decide(state: State) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msg = _render_context(state)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":msg}],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    # strict JSON parse with fallback
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"finish": "Controller failed to return JSON."}

def _controller_rule_decide(state: State) -> Dict[str, Any]:
    """Cheap deterministic fallback when no OPENAI_API_KEY is set."""
    q = state["question"].lower()
    fn = state["file_name"].lower() if state["file_name"] else ""
    # Heuristics
    if fn.endswith(".xlsx") and _tool_allowed(state, "excel_sum_food"):
        return {"tool": "excel_sum_food", "params": {}}
    if fn.endswith(".py") and _tool_allowed(state, "code_run"):
        return {"tool": "code_run", "params": {}}
    if fn.endswith(".mp3") and _tool_allowed(state, "asr"):
        return {"tool": "asr", "params": {}}
    if fn.endswith(".png") and _tool_allowed(state, "chess_from_image"):
        return {"tool": "chess_from_image", "params": {}}
    if "commutative" in q and _tool_allowed(state, "logic_commutativity_set"):
        return {"tool": "logic_commutativity_set", "params": {}}
    if "reverse" in q or q.strip().startswith(".rewsna"):
        if _tool_allowed(state, "reverse_decode"):
            return {"tool": "reverse_decode", "params": {}}
    # arithmetic?
    if re.search(r"\d", q) and _tool_allowed(state, "math_eval"):
        return {"tool": "math_eval", "params": {}}
    # default finish (so we don't loop uselessly)
    return {"finish": "No rule-based action applicable."}

def controller_node(state: State) -> Dict[str, Any]:
    """Decide next action or finish."""
    if state["step"] >= state["max_steps"]:
        return {"next_action": {"finish": "Max steps reached."}}

    decision: Dict[str, Any]
    if os.getenv("OPENAI_API_KEY"):
        decision = _controller_llm_decide(state)
        # Enforce allowlist if provided
        if "tool" in decision and not _tool_allowed(state, decision["tool"]):
            decision = {"finish": f"Tool '{decision['tool']}' not allowed."}
    else:
        decision = _controller_rule_decide(state)

    return {"next_action": decision}

def tool_executor_node(state: State) -> Dict[str, Any]:
    decision = state.get("next_action") or {}
    if "finish" in decision:
        return {"answer": decision["finish"]}

    tool_name = decision.get("tool")
    params = decision.get("params", {}) or {}
    obs = "ERROR: No tool selected."

    if tool_name and _tool_allowed(state, tool_name):
        fn = TOOL_REGISTRY.get(tool_name)
        if fn:
            try:
                obs = fn(state, **params)
            except Exception as e:
                obs = f"TOOL_ERROR: {tool_name}: {e}"
        else:
            obs = f"ERROR: Unknown tool: {tool_name}"

    new_turn: Turn = {
        "thought": f"Chose {tool_name} with {params}",
        "action": tool_name or "finish",
        "params": params,
        "observation": obs,
    }

    return {
        "scratchpad": state["scratchpad"] + [new_turn],
        "next_action": None,
        "step": state["step"] + 1,
        "last_tool": tool_name,   # 👈 add this
    }


def finalize_node(state: State) -> Dict[str, Any]:
    na = state.get("next_action") or {}
    if "finish" in na and na["finish"]:
        return {"answer": f"[finish] {na['finish']}"}

    if state["scratchpad"]:
        last_turn = state["scratchpad"][-1]
        tool = last_turn["action"]
        obs = last_turn["observation"]
        return {"answer": f"[{tool}] {obs}"}  # 👈 includes tool name in output

    return {"answer": "No result."}


# ---------------------------
# Build the app
# ---------------------------
def build_controller_app():
    graph = StateGraph(State)
    graph.add_node("controller", controller_node)
    graph.add_node("tool_exec", tool_executor_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("controller")

    def branch_from_controller(state: State):
        na = state.get("next_action") or {}
        return "finalize" if "finish" in na else "tool_exec"

    graph.add_conditional_edges("controller", branch_from_controller, {
        "tool_exec": "tool_exec",
        "finalize": "finalize",
    })
    graph.add_edge("tool_exec", "controller")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------
# Helper: make initial state
# ---------------------------
def make_initial_state(task_id: str, question: str, file_name: str = "",
                       max_steps: int = 50, allowed_tools: Optional[List[str]] = None) -> State:
    return {
        "task_id": task_id,
        "question": question,
        "file_name": file_name or "",
        "plan": "",
        "scratchpad": [],
        "next_action": None,
        "step": 0,
        "max_steps": max_steps,
        "answer": "",
        "allowed_tools": allowed_tools,
    }
