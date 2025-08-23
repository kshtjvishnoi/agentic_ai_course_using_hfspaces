# agent/tool_exec.py
from typing import Dict, Any
import time, inspect
from .state import State, Turn
from .registry import TOOL_REGISTRY, tool_allowed
from .finalize import early_finish
from agent.finalize import early_finish, normalize_answer



# Add a small alias map (controller sometimes uses these names)
PARAM_ALIASES: dict[str, dict[str, list[str]]] = {
    "math_eval": {"expr": ["expression", "input", "text", "question", "q"]},
    "reverse_decode": {"text": ["input", "question", "q"]},
    "web_search": {"query": ["q", "question", "topic", "prompt", "text", "search"]},
    "wiki_lookup": {"title_or_query": ["query", "q", "title", "page", "topic", "text"]},
    "yt_transcript": {"url": ["video_url", "link"], "video_id": ["vid", "v"]},
    "asr": {"file_path": ["path", "filename", "file", "audio", "mp3"]},
    "chess_from_image": {"file_path": ["path", "filename", "file", "image", "img"]},
    "chess_engine": {"fen": ["position", "FEN", "board"]},
}

def _normalize_params(tool_name: str, fn, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only params the tool accepts; apply alias mapping."""
    raw = raw or {}
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    # If the tool accepts **kwargs, we can pass everything (after alias); else filter to known names
    params = dict(raw)

    # Apply alias mapping (copy first present alias into canonical name if missing)
    aliases = PARAM_ALIASES.get(tool_name, {})
    for canonical, alts in aliases.items():
        if canonical not in params:
            for a in alts:
                if a in raw:
                    params[canonical] = raw[a]
                    break

    if accepts_kwargs:
        return params

    # Otherwise, filter to the function's parameters (excluding the first 'state')
    allowed = set(sig.parameters.keys()) - {"state"}
    return {k: v for k, v in params.items() if k in allowed}


def tool_executor_node(state: State) -> Dict[str, Any]:
    na = state.get("next_action") or {}
    print("tool_exec", na)
    if "finish" in na:
        return {"answer": na["finish"]}

    tool_name = na.get("tool")
    raw_params = na.get("params", {}) or {}
    why = na.get("why", "")
    obs = "ERROR: No tool selected."
    success, err = False, ""
    no_prog = int(state.get("no_progress_count", 0))
    try_same = False

    if not success:
        if state.get("scratchpad"):
            prev = state["scratchpad"][-1]
            prev_action = prev.get("action")
            prev_obs = str(prev.get("observation", ""))
            # If same tool and same error text -> we are stuck repeating
            try_same = (prev_action == tool_name) and (prev_obs == str(obs))
        if try_same:
            no_prog += 1
        else:
            no_prog = 0
    else:
            no_prog = 0
    print("tool_exec", state.get("scratchpad"))

    import time, inspect
    t0 = time.perf_counter()

    if tool_name:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn:
            try:
                params = _normalize_params(tool_name, fn, raw_params)
                obs = fn(state, **params)
                success = not str(obs).startswith(("ERROR", "TOOL_ERROR", "unknown"))
            except Exception as e:
                obs = f"TOOL_ERROR: {tool_name}: {e}"
                err = str(e)
        else:
            obs = f"ERROR: Unknown tool: {tool_name}"
    else:
        obs = "ERROR: Tool not specified."

    dt_ms = int((time.perf_counter() - t0) * 1000)

    new_turn: Turn = {
        "thought": why or f"Chose {tool_name} with {raw_params}",
        "action": tool_name or "finish",
        "params": raw_params,
        "observation": obs,
        "success": success,
        "error": err,
        "duration_ms": dt_ms,
    }

    print("tool_exec", new_turn)

    # basic loop-prevention counter (optional but recommended)
    no_prog = int(state.get("no_progress_count", 0))
    print("tool_exec no prog", no_prog)
    prev_sp = state.get("scratchpad", [])
    print("tool_exec prev_sp", prev_sp)
    if not success and prev_sp:
        prev = prev_sp[-1]
        if prev.get("action") == tool_name and str(prev.get("observation","")) == str(obs):
            no_prog += 1
        else:
            no_prog = 0
    else:
        no_prog = 0 if success else no_prog

    # default next action
    next_action = None
    next_cursor = state.get("plan_cursor", 0) + (1 if success else 0)

    # 👉 EARLY FINISH: if the latest observation seems to fully answer the question
    if success and state.get("early_stop", True):
        should, final_ans = early_finish(state.get("question",""), str(obs), state.get("answer",""))
        if should:
            next_action = {"finish": final_ans, "why": f"Early-stop: confident final answer from {tool_name}."}

    # fallback: if stuck repeating same error 3x, bail out
    if not next_action and not success and no_prog >= 2:
        next_action = {"finish": f"Stopping due to repeated failures calling {tool_name}: {obs}",
                    "why": "loop prevention"}

    print("tool_exec next action", next_action)
    print("tool_exec next cursor", next_cursor)
    print("tool_exec no progress count", no_prog)
    print("tool_exec step", state.get("step", 0) + 1)
    print("tool_exec last tool", tool_name)
    print("tool_exec prev scratchpad", state.get("scratchpad", []))
    return {
        "scratchpad": state["scratchpad"] + [new_turn],
        "next_action": next_action,
        "step": state.get("step", 0) + 1,
        "last_tool": tool_name,
        "plan_cursor": next_cursor,
        "no_progress_count": no_prog,
    }

