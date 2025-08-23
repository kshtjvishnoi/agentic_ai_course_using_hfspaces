from typing import Optional, List
from langgraph.graph import StateGraph, END
from .state import State
from .controller import controller_node
from .tool_exec import tool_executor_node
from .finalize import finalize_node
from .planner import planner_node  
from .config import MAX_STEPS_DEFAULT
from agent.finalize import early_finish, normalize_answer


# Import tools so they register
from . import tools  # noqa: F401

def build_controller_app():
    graph = StateGraph(State)
    graph.add_node("planner", planner_node) 
    graph.add_node("controller", controller_node)
    graph.add_node("tool_exec", tool_executor_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner") 
    graph.add_edge("planner", "controller")

    def branch_from_controller(state: State):
        na = state.get("next_action") or {}
        return "finalize" if "finish" in na else "tool_exec"
    
    

    graph.add_conditional_edges("controller", branch_from_controller, {
        "tool_exec": "tool_exec",
        "finalize": "finalize",
    })
    graph.add_edge("tool_exec", "controller")
    graph.add_edge("finalize", END)
    return graph.compile()

def make_initial_state(task_id: str, question: str, file_name: str = "",
                       max_steps: int = MAX_STEPS_DEFAULT,
                       allowed_tools: Optional[List[str]] = None) -> State:
    return {
        "task_id": task_id,
        "question": question,
        "file_name": file_name or "",
        "plan": [],                 
        "plan_cursor": 0,
        "scratchpad": [],
        "next_action": None,
        "step": 0,
        "max_steps": max_steps,
        "answer": "",
        "allowed_tools": allowed_tools,
        "last_tool": None,
        "auto_finish_after_tool": False,
        "no_progress_count": 0,
        "early_stop": True,     # ADD: enable early finish behavior
        "no_progress_count": 0, # if you haven’t added this before
    }
