"""
Whitelisted tools the agent can call.

Why tools?
  Naive RAG forces every question through vector search exactly once.
  Real questions often need:
    - A second search with a rephrased query (multi-hop)
    - A calculator for "how many minutes is 12 hours?"
    - Combining info from multiple retrieval rounds

Security: only tools in TOOL_REGISTRY are callable — no arbitrary code execution.
"""
import ast
import logging
import operator
from typing import Any, Callable, Dict

from src.core.settings import cfg
from src.ingestion.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# Guard against resource-exhaustion via giant powers, e.g. 9**9**9.
_MAX_POW_EXP = 100
_MAX_MAGNITUDE = 1e100


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > _MAX_POW_EXP or abs(left) > _MAX_MAGNITUDE
        ):
            raise ValueError("operands too large")
        result = _SAFE_OPS[type(node.op)](left, right)
        if abs(result) > _MAX_MAGNITUDE:
            raise ValueError("result too large")
        return result
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic like '12 * 60' or '15 + 30'."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Calculator error: {e}"


def search_docs(query: str, limit: int | None = None) -> str:
    """
    Search ingested documentation — this IS your RAG retriever, but now
    the agent chooses WHEN and HOW OFTEN to call it.
    """
    top_k = limit or cfg("retrieval", "top_k", default=2)
    db = get_vectorstore()
    chunks = db.query_similarity(query, limit=top_k)
    if not chunks:
        return "No relevant documentation found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}]\n{chunk}")
    return "\n\n".join(parts)


def web_search(query: str) -> str:
    """Stub — disabled in production for safety."""
    return (
        f"[web_search stub] Live web search is not configured. "
        f"Query: '{query}'. Use search_docs for local documentation."
    )


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate arithmetic, e.g. '12 * 60' or '100 / 4'",
    },
    "search_docs": {
        "fn": search_docs,
        "description": "Search ingested markdown docs for relevant context",
    },
    "web_search": {
        "fn": web_search,
        "description": "Search the web (stub — not enabled by default)",
    },
}


def get_enabled_tools() -> Dict[str, Callable]:
    enabled = cfg("agentic", "tools", default=["calculator", "search_docs"])
    return {name: TOOL_REGISTRY[name]["fn"] for name in enabled if name in TOOL_REGISTRY}


def tool_descriptions() -> str:
    enabled = cfg("agentic", "tools", default=["calculator", "search_docs"])
    lines = []
    for name in enabled:
        if name in TOOL_REGISTRY:
            lines.append(f"  - {name}: {TOOL_REGISTRY[name]['description']}")
    return "\n".join(lines)
