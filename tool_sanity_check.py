# tool_sanity_check.py
import os
os.environ["PYTHONPATH"] = os.getcwd()  # optional, if running weird shells

from agent import tools  # triggers registration
from agent.registry import TOOL_REGISTRY

print("Registered tools:", list(TOOL_REGISTRY.keys()))

# Optional: call openai_answer directly once
from agent.tools.openai_engine import openai_answer_tool

dummy_state = {
    "task_id": "t_demo",
    "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
    "file_name": "",
}
print("openai_answer output:", openai_answer_tool(dummy_state)[:120])
