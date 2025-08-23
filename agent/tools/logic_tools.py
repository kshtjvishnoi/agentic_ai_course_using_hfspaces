import re
from ..registry import tool
from ..state import State


@tool("math_eval")
def math_eval_tool(state: State, expr: str | None = None, **kwargs) -> str:
    """
    Safely evaluate arithmetic. Accepts aliases: expression/input/text/question/q via executor aliasing,
    but also checks kwargs as a second layer of robustness.
    """
    import ast, operator as op
    if expr is None:
        expr = (
            kwargs.get("expression")
            or kwargs.get("input")
            or kwargs.get("text")
            or kwargs.get("question")
            or kwargs.get("q")
            or state["question"]
        )
    s = str(expr).replace("=", "").strip()
    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
               ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
               ast.FloorDiv: op.floordiv, ast.Mod: op.mod}
    def ev(n):
        if isinstance(n, ast.Num): return n.n
        if isinstance(n, ast.UnaryOp) and type(n.op) in [ast.USub]: return allowed[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in allowed: return allowed[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("Unsupported")
    try:
        return str(ev(ast.parse(s, mode="eval").body))
    except Exception as e:
        return f"ERROR: math_eval: {e}"

