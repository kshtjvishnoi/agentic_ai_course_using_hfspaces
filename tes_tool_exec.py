# test_tool_exec.py
import argparse, json, sys, traceback, importlib, inspect, re, os
from typing import Any, Dict

def find_controller_fn():
    try:
        ctrl = importlib.import_module("agent.controller")
    except Exception:
        return None
    candidates = [
        "controller_node", "controller_step", "choose_next_action",
        "decide_next_action", "controller", "run_controller",
        "run", "next_action",
    ]
    for name in candidates:
        fn = getattr(ctrl, name, None)
        if callable(fn):
            try:
                if len(inspect.signature(fn).parameters) == 1:
                    return fn
            except Exception:
                return fn
    return None

def parse_kv_pairs(pairs: list[str]) -> Dict[str, Any]:
    """Parse repeated -p/--param KEY=VALUE entries."""
    result: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid -p/--param entry (need key=value): {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        # strip one layer of quotes if present
        if (len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'"))):
            v = v[1:-1]
        result[k] = v
    return result

def _quote_bare_object_keys(s: str) -> str:
    # Turn {expr:2+2, foo:"bar"} into {"expr":2+2, "foo":"bar"}
    def repl(m):
        return f'{m.group(1)}"{m.group(2)}"{m.group(3)}'
    return re.sub(r'([{\s,])\s*([A-Za-z_][\w-]*)\s*(:)', repl, s)

def _parse_ps_hashtable(s: str) -> Dict[str, Any]:
    # Parse PowerShell hashtable like: @{ expr = "2+2"; x = 5 }
    inner = s.strip()
    if inner.startswith("@{"):
        inner = inner[2:]
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    out: Dict[str, Any] = {}
    # split on ; but allow ; to be optional (newline/space separated)
    # find key = value pairs
    for m in re.finditer(r'([A-Za-z_][\w-]*)\s*=\s*(".*?"|\'.*?\'|[^;]+)', inner, flags=re.DOTALL):
        k = m.group(1)
        v = m.group(2).strip()
        if (len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'"))):
            v = v[1:-1]
        out[k] = v.strip()
    return out

def coerce_params(args) -> Dict[str, Any]:
    # Highest priority: -p/--param key=value (repeatable)
    if args.param:
        return parse_kv_pairs(args.param)

    # File input
    if args.params_file:
        with open(args.params_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # STDIN
    if args.params == "-":
        data = sys.stdin.read()
        return json.loads(data)

    # Raw --params
    if args.params:
        raw = args.params.strip()
        # Try strict JSON first
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Try PowerShell hashtable syntax: @{ expr="2+2"; x=5 }
        if raw.startswith("@{"):
            try:
                return _parse_ps_hashtable(raw)
            except Exception:
                pass
        # Try bare-object-keys like {expr:2+2}
        if raw.startswith("{") and ":" in raw:
            try:
                return json.loads(_quote_bare_object_keys(raw))
            except Exception:
                pass
        # Try single key=value form
        if "=" in raw and " " not in raw and "{" not in raw and "}" not in raw:
            return parse_kv_pairs([raw])

        raise ValueError("Could not parse --params as JSON/hashtable/kv.")

    return {}

def main():
    parser = argparse.ArgumentParser(description="Test runner for agent/tool_exec.py (robust params parsing)")
    parser.add_argument("--question", type=str, default="", help="User question (controller-driven mode if controller exists)")
    parser.add_argument("--tool", type=str, default=None, help="Tool name for manual tool call")
    parser.add_argument("--params", type=str, default=None,
                        help='Params as JSON, PowerShell hashtable (@{k="v"}), {k:v}, or single kv like expr=2+2. Use "-" to read JSON from STDIN.')
    parser.add_argument("--param", "-p", action="append", default=[],
                        help="Repeatable key=value param, e.g. -p expr=2+2 -p base=10")
    parser.add_argument("--params-file", type=str, default=None, help="Path to JSON file with params.")
    parser.add_argument("--max-steps", type=int, default=6, help="Max agent steps")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early_finish heuristic")
    args = parser.parse_args()

    # Import real executor & registry
    try:
        from agent.tool_exec import tool_executor_node
        from agent.registry import TOOL_REGISTRY
    except Exception as e:
        print("ERROR: Failed to import agent modules. Are you running from repo root?\n", e, file=sys.stderr)
        sys.exit(1)

    # Ensure tools package is imported so decorators register tools
    try:
        importlib.import_module("agent.tools")
    except Exception:
        pass

    # Build initial state
    state: Dict[str, Any] = {
        "question": args.question or "",
        "scratchpad": [],
        "next_action": None,
        "plan_cursor": 0,
        "step": 0,
        "no_progress_count": 0,
        "early_stop": (not args.no_early_stop),
        "answer": "",
    }

    # Seed next_action
    if args.tool:
        try:
            parsed_params = coerce_params(args)
        except Exception as e:
            print(f"ERROR parsing params: {e}", file=sys.stderr)
            sys.exit(2)

        if args.param or args.params or args.params_file:
            print("DEBUG parsed params:", parsed_params)

        if args.tool not in TOOL_REGISTRY:
            print(f"ERROR: Tool '{args.tool}' not found. Available: {sorted(TOOL_REGISTRY.keys())}", file=sys.stderr)
            sys.exit(3)

        state["next_action"] = {
            "tool": args.tool,
            "params": parsed_params,
            "why": "Manual test via CLI",
        }
    else:
        controller_fn = find_controller_fn()
        if not controller_fn:
            print("ERROR: No --tool provided and no controller function found in agent.controller.", file=sys.stderr)
            print("Tip: manual mode, e.g.:", file=sys.stderr)
            print(r"  python .\test_tool_exec.py --tool math_eval -p expr=2+2", file=sys.stderr)
            sys.exit(4)
        try:
            na = controller_fn(state)
            if not isinstance(na, dict) or ("tool" not in na and "finish" not in na):
                print("ERROR: Controller did not return a valid next_action dict.\nGot:", na, file=sys.stderr)
                sys.exit(5)
            state["next_action"] = na
        except Exception:
            print("ERROR: Calling your real controller failed:\n", traceback.format_exc(), file=sys.stderr)
            sys.exit(6)

    def merge(diff: Dict[str, Any]):
        for k, v in diff.items():
            if k == "scratchpad":
                state["scratchpad"] = v
            else:
                state[k] = v

    # Run loop
    for _ in range(args.max_steps):
        print("\n=== STEP", state.get("step", 0), "===")
        na = state.get("next_action") or {}
        if "finish" in na:
            print("FINISH (from controller):", na["finish"])
            state["answer"] = na["finish"]
            break

        diff = tool_executor_node(state)
        merge(diff)

        if "answer" in diff and diff["answer"]:
            print("FINISH (from tool_exec early finish):", diff["answer"])
            break

        if state.get("scratchpad"):
            last = state["scratchpad"][-1]
            print("Action :", last.get("action"))
            print("Params :", last.get("params"))
            print("Success:", last.get("success"))
            print("Obs    :", (str(last.get("observation")) or "")[:1000])
            print("Why    :", last.get("thought"))
            print("Dur ms :", last.get("duration_ms"))

        na = state.get("next_action")
        if na and "finish" in na:
            print("FINISH (from tool_exec):", na["finish"])
            state["answer"] = na["finish"]
            break
        if not na:
            controller_fn = find_controller_fn()
            if not controller_fn:
                print("No next_action and no controller found. Stopping.", file=sys.stderr)
                break
            try:
                na2 = controller_fn(state)
                if not isinstance(na2, dict) or ("tool" not in na2 and "finish" not in na2):
                    print("Controller did not return a valid next_action dict. Stopping.", file=sys.stderr)
                    break
                state["next_action"] = na2
            except Exception:
                print("ERROR: Controller call failed:\n", traceback.format_exc(), file=sys.stderr)
                break

    print("\n=== FINAL STATE ===")
    print("Steps        :", state.get("step"))
    print("Plan cursor  :", state.get("plan_cursor"))
    print("No-progress  :", state.get("no_progress_count"))
    print("Answer       :", state.get("answer"))
    print("Last tool    :", state.get("last_tool"))
    print("Turns logged :", len(state.get("scratchpad", [])))

if __name__ == "__main__":
    main()
