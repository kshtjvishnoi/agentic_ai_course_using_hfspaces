import os, sys, json, csv, time, pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional

# project imports
from agent.graph import build_controller_app, make_initial_state

# ---- optional hub download (gated dataset) ----
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF_HUB = True
except Exception:
    _HAS_HF_HUB = False

DATA_ROOT = pathlib.Path("data")           # where we cache files we download
RUNS_ROOT = pathlib.Path("runs")           # where we save outputs
RUNS_ROOT.mkdir(exist_ok=True, parents=True)
DATA_ROOT.mkdir(exist_ok=True, parents=True)

GAIA_REPO = "gaia-benchmark/GAIA"          # dataset repo (gated)
GAIA_PREFIXES = [
    "2023/validation",                     # most Level 1 assets live here
    "2023/test", "2023/train"              # fallbacks if needed
]

def resolve_file(file_name: str) -> Optional[str]:
    """
    Try to locate the attachment locally; if missing and HF hub is available,
    try to download into ./data/<prefix>/<file_name>.
    """
    if not file_name:
        return None

    # 1) If absolute path or relative file exists
    p = pathlib.Path(file_name)
    if p.exists():
        return str(p.resolve())

    # 2) If in ./data anywhere
    for path in DATA_ROOT.rglob(file_name):
        if path.name == file_name:
            return str(path.resolve())

    # 3) Try to download from HF hub (if available)
    if _HAS_HF_HUB and os.getenv("HUGGINGFACE_HUB_TOKEN"):
        for prefix in GAIA_PREFIXES:
            hub_path = f"{prefix}/{file_name}"
            try:
                local = hf_hub_download(
                    repo_id=GAIA_REPO,
                    repo_type="dataset",
                    filename=hub_path,
                    local_dir=str(DATA_ROOT),
                )
                return local
            except Exception:
                continue

    # 4) Not found
    return None

def run_batch(questions: List[Dict[str, Any]], max_steps: int = 8) -> Dict[str, Any]:
    app = build_controller_app()
    results = []
    traces_path = RUNS_ROOT / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    n_ok = n_err = 0

    with open(traces_path, "w", encoding="utf-8") as ftrace:
        for i, item in enumerate(questions, 1):
            task_id = item.get("task_id", f"task_{i}")
            question = item.get("question", "").strip()
            file_name = item.get("file_name", "").strip()

            resolved = resolve_file(file_name) if file_name else None

            state = make_initial_state(
                task_id=task_id,
                question=question,
                file_name=resolved or "",          # give absolute path (or empty)
                max_steps=max_steps,
                # no allowlist: the controller will choose tools
            )
            # multi-step so planner/controller can chain (e.g., chess_from_image -> chess_engine)
            state["auto_finish_after_tool"] = False

            t0 = time.perf_counter()
            out = app.invoke(state)
            dt = int((time.perf_counter() - t0) * 1000)

            answer = out.get("answer", "")
            last_tool = out.get("last_tool", "")
            ok = not (isinstance(answer, str) and answer.lower().startswith(("error", "tool_error")))
            n_ok += 1 if ok else 0
            n_err += 0 if ok else 1

            results.append({
                "task_id": task_id,
                "question": question,
                "file_name": file_name,
                "resolved_file": resolved or "",
                "answer": answer,
                "last_tool": last_tool,
                "duration_ms": dt,
                "steps": len(out.get("scratchpad", [])),
            })

            # write full trace (one line per task)
            json.dump({
                "task_id": task_id,
                "state_out": out,       # includes plan, scratchpad, etc.
            }, ftrace, ensure_ascii=False)
            ftrace.write("\n")

            print(f"[{i}/{len(questions)}] {task_id} → "
                  f"{'OK' if ok else 'ERR'} in {dt}ms | tool={last_tool} | file={'Y' if resolved else 'N'}")

    # CSV summary
    csv_path = RUNS_ROOT / "run_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    return {
        "results_csv": str(csv_path.resolve()),
        "traces_jsonl": str(traces_path.resolve()),
        "n_ok": n_ok,
        "n_err": n_err,
        "n_total": len(questions),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_questions.py <questions.json>")
        sys.exit(1)

    qfile = pathlib.Path(sys.argv[1]).resolve()
    with open(qfile, "r", encoding="utf-8") as f:
        questions = json.load(f)

    summary = run_batch(questions, max_steps=8)
    print("\n== Summary ==")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
