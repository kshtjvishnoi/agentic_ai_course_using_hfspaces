from typing import TypedDict, List, Dict, Any, Optional

class Turn(TypedDict):
    thought: str                # short “why” the controller chose this step
    action: str                 # tool name (e.g., "reverse_decode")
    params: Dict[str, Any]      # inputs provided to the tool
    observation: str            # tool output or error
    success: bool               # optional: whether the tool step succeeded
    error: str                  # optional: error text if any
    duration_ms: int            # optional: runtime of the tool

class PlanStep(TypedDict, total=False):
    goal: str               # sub-goal for this step
    tool_hint: str          # suggested tool name (may be empty)
    params_hint: Dict[str, Any]  # suggested params (optional)

class State(TypedDict, total=False):
    task_id: str
    question: str
    file_name: str
    plan: List[PlanStep]
    plan_cursor: int
    scratchpad: List[Turn]
    next_action: Optional[Dict[str, Any]]   # {"tool":..., "params":..., "why": "..."} OR {"finish": "...", "why":"..."}
    step: int
    max_steps: int
    answer: str
    allowed_tools: Optional[List[str]]
    last_tool: Optional[str]
    auto_finish_after_tool: bool
    no_progress_count: int
