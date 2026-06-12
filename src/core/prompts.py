from pathlib import Path

from src.core.settings import ROOT_DIR, cfg

DEFAULT_AGENT_PROMPT = """You are a documentation assistant. Use search_docs to find context.
Call tools with: TOOL: <name> ARGS: <args>"""


def prompts_dir() -> Path:
    rel = cfg("paths", "prompts_dir", default="prompts")
    return ROOT_DIR / rel


def prompt_version() -> str:
    return cfg("prompts", "version", default="v1")


def load_prompt(name: str, version: str | None = None) -> str:
    ver = version or prompt_version()
    path = prompts_dir() / ver / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    if name == "agent_system":
        return DEFAULT_AGENT_PROMPT
    raise FileNotFoundError(f"Prompt not found: {path}")
