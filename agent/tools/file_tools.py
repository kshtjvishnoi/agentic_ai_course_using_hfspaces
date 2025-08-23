from ..registry import tool
from ..state import State

@tool("code_run")
def code_run_tool(state: State, file_path: str | None = None, timeout: int = 5) -> str:
    import tempfile, subprocess
    path = file_path or state["file_name"]
    if not path or not path.endswith(".py"):
        return "ERROR: No .py file."
    try:
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(["python", "-I", "-B", path], cwd=td,
                                  capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return f"ERROR: code_run: {proc.stderr.strip()[:400]}"
        out = proc.stdout.strip().splitlines()
        return out[-1].strip() if out else ""
    except Exception as e:
        return f"ERROR: code_run: {e}"
