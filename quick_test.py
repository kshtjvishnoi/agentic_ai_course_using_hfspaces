import os
import io
import time
import json
import shutil
import requests
import inspect
import pathlib
import traceback
import pandas as pd
import gradio as gr
from pathlib import Path
import subprocess, socket


DEBUG = os.getenv("DEBUG_AGENT", "1") == "1"

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def _short(s, n=180):
    s = str(s or "").replace("\n", " ").replace("\r", " ")
    return (s[:n] + "…") if len(s) > n else s

# ====== keep constants ======
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# ====== your agent stack ======
# These come from your modular codebase we built earlier
from agent.graph import build_controller_app, make_initial_state

# Optional: gated GAIA asset download from the Hub (if token available)
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF_HUB = True
except Exception:
    _HAS_HF_HUB = False

# ---------- paths ----------
DATA_ROOT = pathlib.Path("data")
RUNS_ROOT = pathlib.Path("runs")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_PREFIXES = [
    "2023/validation",
    "2023/test",
    "2023/train",
]

# ---------- helpers ----------

def _default_agent_code() -> str:
    sid = os.getenv("SPACE_ID")
    if sid:
        return f"https://huggingface.co/spaces/{sid}/tree/main"
    # try git remote
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        if url:
            if url.startswith("git@"):
                host, path = url.split(":", 1)
                host = host.replace("git@", "")
                url = f"https://{host}/{path}"
            return url.removesuffix(".git")
    except Exception:
        pass
    # long local fallback to satisfy API min-length
    return f"local://{socket.gethostname()}{Path.cwd()}"


def _resolve_attachment_local_or_download(api_url: str, task: dict) -> str:
    """
    Try to resolve task['file_name'] to a local path.
    Resolution order:
      1) Absolute or relative existing path
      2) ./data/** recursively
      3) HF Space file endpoint: GET {api_url}/files/{task_id}
      4) HF Hub dataset download (needs HUGGINGFACE_HUB_TOKEN)
    Returns absolute path string or "".
    """
    file_name = (task.get("file_name") or "").strip()
    if not file_name:
        return ""

    # 1) local relative/absolute
    fpath = pathlib.Path(file_name)
    if fpath.exists():
        return str(fpath.resolve())

    # 2) search under ./data
    for p in DATA_ROOT.rglob(file_name):
        if p.name == file_name:
            return str(p.resolve())

    # 3) try Space file endpoint (works when running inside HF Space)
    try:
        files_url = f"{api_url.rstrip('/')}/files/{task['task_id']}"
        r = requests.get(files_url, stream=True, timeout=30)
        if r.status_code == 200:
            target = DATA_ROOT / file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return str(target.resolve())
    except Exception:
        pass  # fall through

    # 4) try HF Hub (gated dataset)
    if _HAS_HF_HUB and os.getenv("HUGGINGFACE_HUB_TOKEN"):
        for prefix in GAIA_PREFIXES:
            try:
                local = hf_hub_download(
                    repo_id=GAIA_REPO,
                    repo_type="dataset",
                    filename=f"{prefix}/{file_name}",
                    local_dir=str(DATA_ROOT),
                )
                return str(pathlib.Path(local).resolve())
            except Exception:
                continue

    return ""


def _build_reasoning_trace(out_state: dict) -> str:
    """
    Turn plan + scratchpad into a compact text trace for the competition JSONL.
    """
    lines = []
    plan = out_state.get("plan") or []
    if plan:
        lines.append("PLAN:")
        for i, step in enumerate(plan, 1):
            lines.append(f"  {i}. goal={step.get('goal','')} | tool_hint={step.get('tool_hint','')} | params_hint={step.get('params_hint',{})}")

    sp = out_state.get("scratchpad") or []
    if sp:
        lines.append("TRACE:")
        for i, t in enumerate(sp, 1):
            action = t.get("action", "")
            why = t.get("thought", "")
            params = t.get("params", {})
            obs = str(t.get("observation", ""))[:300].replace("\n", " ")
            ok = t.get("success", False)
            lines.append(f"  {i}. [{action}] {why} | params={params} | ok={ok} | obs={obs}")
    # Final answer (as seen by finalize node)
    ans = out_state.get("answer", "")
    if ans:
        lines.append(f"FINAL: {ans}")
    return "\n".join(lines)


class ControllerAgent:
    """
    Wraps your LangGraph controller. Call with (question, file_path, task_id) and returns (answer, trace, last_tool, steps, elapsed_ms).
    """
    def __init__(self):
        self.app = build_controller_app()
        print("ControllerAgent initialized.")

    def __call__(self, *, task_id: str, question: str, file_path: str = "", max_steps: int = 50):
        state = make_initial_state(
            task_id=task_id,
            question=question,
            file_name=file_path,
            max_steps=max_steps,
        )
        # multi-step by default so planner/controller can chain tools
        state["auto_finish_after_tool"] = False

        t0 = time.perf_counter()
        rec_lim = max(120, 5 * max_steps)  # e.g., 20 steps → 100; cap to >=120
        dprint(f"[RUN] {task_id}: max_steps={max_steps} recursion_limit={rec_lim}")
        out = self.app.invoke(state, config={"recursion_limit": rec_lim})
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        answer = out.get("answer", "")
        reasoning_trace = _build_reasoning_trace(out)
        last_tool = out.get("last_tool", "")
        steps = len(out.get("scratchpad", []))
        return answer, reasoning_trace, last_tool, steps, elapsed_ms


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
    agent_code_override: str = "",
    max_steps: int = 20,
):
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (ControllerAgent preferred; fallback to BasicAgent)
    try:
        from agent.graph import build_controller_app, make_initial_state  # if not already imported
        class ControllerAgent:
            def __init__(self):
                self.app = build_controller_app()
            def __call__(self, *, task_id: str, question: str, file_path: str = "", max_steps: int = 20):
                state = make_initial_state(task_id=task_id, question=question, file_name=file_path, max_steps=max_steps)
                state["auto_finish_after_tool"] = False
                rec_lim = max(120, 5 * max_steps)
                dprint(f"[RUN] {task_id}: max_steps={max_steps} recursion_limit={rec_lim}")
                out = self.app.invoke(state, config={"recursion_limit": rec_lim})
                # Build a simple reasoning trace from plan + scratchpad (if available)
                plan = out.get("plan") or []
                sp = out.get("scratchpad") or []
                lines = []
                if plan:
                    lines.append("PLAN:")
                    for i, s in enumerate(plan, 1):
                        lines.append(f"  {i}. goal={s.get('goal','')} | tool_hint={s.get('tool_hint','')}")
                if sp:
                    lines.append("TRACE:")
                    for i, t in enumerate(sp, 1):
                        lines.append(f"  {i}. [{t.get('action','')}] {t.get('thought','')} → {str(t.get('observation',''))[:200]}")
                trace = "\n".join(lines)

                        # Plan summary
                if plan:
                    dprint(f"[PLAN] {task_id}: {len(plan)} steps")
                    for i, ps in enumerate(plan, 1):
                        dprint(f"  {i}. goal={ps.get('goal','')} | tool_hint={ps.get('tool_hint','')} | params_hint={ps.get('params_hint',{})}")
                # Scratchpad steps
                if sp:
                    dprint(f"[TRACE] {task_id}: {len(sp)} turns")
                    for i, t in enumerate(sp, 1):
                        dprint(f"  [STEP {i}] action={t.get('action','')} success={t.get('success',False)} dur={t.get('duration_ms','?')}ms")
                        dprint(f"            params={t.get('params',{})}")
                        dprint(f"            obs={_short(t.get('observation',''), 200)}")

                dprint(f"[ANSWER] {task_id}: {_short(answer, 200)} | last_tool={last_tool}")
                dprint("-"*80)

                return out.get("answer",""), trace, out.get("last_tool",""), len(sp), 0
        agent = ControllerAgent()
    except Exception as e:
        print(f"Error instantiating ControllerAgent: {e}")
        return f"Error initializing agent: {e}", None

    # compute agent_code (use override if provided)
    agent_code = (agent_code_override or "").strip() or _default_agent_code()
    if len(agent_code) < 10:
        agent_code = _default_agent_code()
    print("agent_code:", agent_code)

    # 2. Fetch Questions (unchanged)
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=30)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding server response for questions: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run agent
    results_log, answers_payload = [], []
    print(f"Running agent on {len(questions_data)} questions...")
    for idx, item in enumerate(questions_data, 1):
        
        task_id = item.get("task_id") or f"task_{idx}"
        question_text = item.get("question") or ""             # always defined
        file_path = (item.get("file_name") or "").strip()

        dprint(f"[TASK {idx}/{len(questions_data)}] id={task_id}")
        dprint(f"[Q] {_short(question_text, 240)}")
        dprint(f"[FILE] {file_path or '-'}")
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            file_path = (item.get("file_name") or "").strip()
            # If you have a resolver/downloader, call it here to turn file_name into an absolute path.
            answer, trace, last_tool, steps, elapsed_ms = agent(
                task_id=task_id, question=question_text, file_path=file_path, max_steps=max_steps
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            answer, trace, last_tool, steps, elapsed_ms = f"AGENT ERROR: {e}", "", "", 0, 0

        answers_payload.append({"task_id": task_id, "submitted_answer": answer})
        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Resolved File": file_path,
            "Submitted Answer": answer,
            "Tool Used": last_tool,
            "Steps": steps,
            "Duration (ms)": elapsed_ms,
        })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit (unchanged)
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# ====== UI ======
with gr.Blocks() as demo:
    gr.Markdown("# Agentic Controller Runner (LangGraph + Tools)")
    gr.Markdown(
        """
        **Instructions**
        1. Log in (OAuth) with your Hugging Face account.
        2. Click **Run Evaluation & Submit All Answers**.
        3. The app will fetch the questions, run your controller on each, submit answers, and save a JSONL for competition.

        **Notes**
        - Attached files (PNG/MP3/XLSX/…) are auto-resolved from the Space or GAIA Hub if your token is set.
        - A JSONL is saved to `runs/competition_output.jsonl` with:
          `{"task_id": "...", "model_answer": "...", "reasoning_trace": "..."}` one per line.
        """
    )

    gr.LoginButton()
    agent_code_box = gr.Textbox(
    label="Agent code URL (repo link)",
    value=_default_agent_code(),
    placeholder="https://huggingface.co/spaces/<user>/<space>/tree/main or your GitHub repo URL",)

    max_steps_inp = gr.Slider(
    minimum=5, maximum=120, step=1, value=40,
    label="Max Steps",)


    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=10, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    max_steps_inp = gr.Slider( minimum=5, maximum=60, step=1, value=20, label="Max Steps" )

    run_button.click(fn=run_and_submit_all, inputs=[agent_code_box, max_steps_inp],  outputs=[status_output, results_table])


if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID not found (running locally?).")

    print("-"*(60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface…")
    demo.launch(debug=True, share=False)
