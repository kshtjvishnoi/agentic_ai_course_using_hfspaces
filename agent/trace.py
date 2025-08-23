import json, time
from .state import State

def save_turns_jsonl(state: State, path: str = "runs.jsonl") -> None:
    record = {
        "task_id": state["task_id"],
        "question": state["question"],
        "file_name": state.get("file_name"),
        "turns": state["scratchpad"],
        "final_answer": state.get("answer",""),
        "last_tool": state.get("last_tool"),
        "ts": time.time(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def format_trace(state: State) -> str:
    lines = []
    for i, t in enumerate(state["scratchpad"], 1):
        lines.append(
            f"{i}. [{t.get('action','?')}] "
            f"{t.get('thought','')}\n"
            f"   params={t.get('params',{})} | "
            f"duration={t.get('duration_ms','?')}ms | "
            f"success={t.get('success','?')}\n"
            f"   → {t.get('observation','')}\n"
        )
    return "\n".join(lines)
