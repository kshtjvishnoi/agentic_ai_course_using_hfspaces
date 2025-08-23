# quick_test_planner.py
import argparse
import json
import sys

# Ensure the package is importable when running from repo root
# (If your project already installs as a package, you can drop this.)
from pathlib import Path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

# Import the planner and types
from agent.planner import planner_node
from agent.state import State

# Importing tools package to trigger tool registration side-effects
# (tools/__init__.py should import all tool modules to fill TOOL_REGISTRY)
try:
    import agent.tools  # noqa: F401
except Exception as e:
    print(f"Warning: couldn't import agent.tools for tool registration: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run the planning node with your question.")
    parser.add_argument("-q", "--question", help="Question/task to plan for")
    parser.add_argument("-f", "--file", dest="file_name", help="Attached filename hint (optional)")
    args = parser.parse_args()

    question = args.question or input("Enter your question to plan for: ").strip()
    state: State = {"question": question}
    if args.file_name:
        state["file_name"] = args.file_name

    # Call your actual planner node (no mocks)
    result = planner_node(state)

    # Pretty print the plan
    print("\n--- Planner Output ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Optional: quick visibility of config values
    try:
        from agent.config import OPENAI_API_KEY, OPENAI_MODEL
        if not OPENAI_API_KEY:
            print("Note: OPENAI_API_KEY is empty in agent.config; planner will return an empty plan.")
        else:
            print(f"Using model from agent.config: {OPENAI_MODEL}")
    except Exception as e:
        print(f"Note: couldn't read agent.config: {e}")

    main()
