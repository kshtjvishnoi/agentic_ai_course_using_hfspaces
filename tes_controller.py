# run_controller.py
"""
Runs your real controller against your real tool registry.

Usage examples:
  python run_controller.py --question "Summarize the file" --file-name report.pdf
  python run_controller.py --question "Compute KPI from csv" --max-steps 6
  python run_controller.py --question "..." --auto-finish

Notes:
- Imports controller_node from your code and calls it directly.
- Executes tools from your TOOL_REGISTRY.
- Respects tool_allowed() if defined to gate execution.
"""

import argparse
import json
import traceback

from agent.controller import controller_node
from agent.registry import TOOL_REGISTRY, tool_allowed


def run_tool(tool_name: str, params: dict, state: dict) -> dict:
    """Execute a real tool from your registry and return a uniform result dict."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"ok": False, "observation": f"Unknown tool: {tool_name}"}

    # Gate via allowlist if your project uses it
    try:
        if callable(tool_allowed) and not tool_allowed(tool_name, state):
            return {"ok": False, "observation": f"Tool not allowed: {tool_name}"}
    except TypeError:
        # Back-compat: legacy signatures like tool_allowed(tool_name)
        if not tool_allowed(tool_name):
            return {"ok": False, "observation": f"Tool not allowed: {tool_name}"}

    try:
        result = fn(**(params or {}))
        return {"ok": True, "observation": result}
    except TypeError as e:
        return {"ok": False, "observation": f"Bad params for {tool_name}: {e}"}
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return {"ok": False, "observation": f"Tool error: {e}\n{tb}"}


def make_initial_state(args: argparse.Namespace) -> dict:
    """Seed a minimal state your controller expects. Edit as needed for your app."""
    plan = []
    if args.plan:
        # plan can be a JSON string or @path/to/plan.json
        raw = args.plan
        if raw.startswith("@"):
            with open(raw[1:], "r", encoding="utf-8") as f:
                plan = json.load(f)
        else:
            plan = json.loads(raw)

    return {
        "question": args.question,
        "file_name": args.file_name,
        "plan": plan,                  # optional; controller handles empty plan
        "plan_cursor": 0,
        "scratchpad": [],              # list of {thought, action, params, observation, success}
        "step": 0,
        "max_steps": args.max_steps,
        "auto_finish_after_tool": args.auto_finish,
    }


def pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Task/question for the controller.")
    ap.add_argument("--file-name", default=None, help="Optional file name/path to include in state.")
    ap.add_argument("--plan", default="", help="JSON plan or @path/to/plan.json (optional).")
    ap.add_argument("--max-steps", type=int, default=8, help="Safety cap on controller loops.")
    ap.add_argument("--auto-finish", action="store_true",
                    help="If last tool succeeded, let controller finish early.")
    args = ap.parse_args()

    state = make_initial_state(args)

    print("=== RUN START ===")
    print("Question:", state["question"])
    if state.get("file_name"):
        print("File:", state["file_name"])
    if state.get("plan"):
        print("Plan loaded with", len(state["plan"]), "steps")

    # Main controller loop
    for i in range(args.max_steps + 2):  # tiny cushion beyond max_steps
        out = controller_node(state)  # << CALLS YOUR REAL controller.py FUNCTION
        decision = (out or {}).get("next_action", {})

        print(f"\n[Step {state.get('step', 0)}] Controller decision:")
        print(pretty(decision))

        # Finish?
        if "finish" in decision:
            print("\n[FINAL ANSWER]")
            print(decision.get("finish", ""))
            break

        # Tool call?
        tool = decision.get("tool")
        params = decision.get("params") or {}
        why = decision.get("why") or ""

        if not tool:
            print("\n[WARN] Controller did not return a tool or finish; stopping.")
            break

        # Execute the real tool from your registry
        result = run_tool(tool, params, state)
        print("\n[Tool execution]")
        print(f"tool={tool} params={pretty(params)} -> success={result['ok']}")
        print("observation:", pretty(result["observation"]))

        # Append to scratchpad and advance
        state.setdefault("scratchpad", []).append({
            "thought": why,
            "action": tool,
            "params": params,
            "observation": result["observation"],
            "success": bool(result["ok"]),
        })
        state["step"] = state.get("step", 0) + 1

        # Naive plan cursor advance (edit if your plan logic differs)
        if isinstance(state.get("plan"), list) and state["plan"]:
            state["plan_cursor"] = min(state.get("plan_cursor", 0) + 1, len(state["plan"]) - 1)

    else:
        print("\n[Loop cap reached without finish]")

    print("\n=== RUN END ===")


if __name__ == "__main__":
    main()
