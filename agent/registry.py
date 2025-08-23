from typing import Callable, Dict
from .state import State

ToolFn = Callable[..., str]
TOOL_REGISTRY: Dict[str, ToolFn] = {}

def tool(name: str):
    def deco(fn: ToolFn):
        TOOL_REGISTRY[name] = fn
        return fn
    return deco

def tool_allowed(state: State, name: str) -> bool:
    return name in TOOL_REGISTRY and (not state.get("allowed_tools") or name in state["allowed_tools"])
