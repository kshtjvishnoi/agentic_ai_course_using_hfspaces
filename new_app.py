from agent.graph import build_controller_app, make_initial_state
from agent.trace import format_trace  # optional for a separate “Trace” textbox

controller_app = build_controller_app()

def run_agent_controller(task):
    s = make_initial_state(
        task_id=task["task_id"],
        question=task["question"],
        file_name=task.get("file_name", "") or "",
        max_steps=6,
    )
    out = controller_app.invoke(s)
    # Pull last tool + why
    last_tool = out.get("last_tool") or ""
    why = ""
    if out.get("scratchpad"):
        why = out["scratchpad"][-1].get("thought", "")
    answer = out.get("answer", "")
    trace_md = format_trace(out)  # optional, to show a full step-by-step trace
    return answer, last_tool, why, trace_md
