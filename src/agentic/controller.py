
import logging
import re
from typing import Any, Dict, List, TypedDict

from ollama import chat

from src.core.prompts import load_prompt
from src.core.settings import cfg
from src.agentic.tools import get_enabled_tools, tool_descriptions
from src.agentic.guardrails import check_answer
from src.observability.tracing import Tracer

logger = logging.getLogger(__name__)

TOOL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*ARGS:\s*(.+)", re.IGNORECASE | re.DOTALL)


class AgentResult(TypedDict, total=False):
    """Shape of the dict returned by AgenticController.run().

    A TypedDict documents the contract for callers (API, evaluator, CLI) without
    changing runtime behavior — it's still a plain dict.
    """

    answer: str
    steps: int
    tool_calls: List[Dict[str, str]]
    retrieved_contexts: List[str]
    guardrails: Dict[str, Any]
    tools_used: List[str]


class AgenticController:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or cfg("models", "llm", default="llama3.2:3b")
        self.max_steps = cfg("agentic", "max_steps", default=5)
        self.tools = get_enabled_tools()

    def _system_prompt(self) -> str:
        base = load_prompt("agent_system")
        return f"{base}\n\nAvailable tools:\n{tool_descriptions()}"

    def _parse_tool_call(self, text: str) -> tuple[str | None, str | None]:
        match = TOOL_PATTERN.search(text)
        if match:
            return match.group(1).lower().strip(), match.group(2).strip()
        return None, None

    def _is_final_answer(self, text: str) -> bool:
        """If the LLM didn't request a tool, treat the message as the final answer."""
        tool_name, _ = self._parse_tool_call(text)
        return tool_name is None

    def _finalize(self, result: AgentResult) -> AgentResult:
        """Attach guardrail verdict + tool-usage summary to a result dict."""
        verdict = check_answer(result["answer"], result.get("retrieved_contexts", []))
        result["guardrails"] = verdict.to_dict()
        result["tools_used"] = sorted({c["tool"] for c in result.get("tool_calls", [])})
        return result

    def run(self, question: str) -> AgentResult:
        # Trace the whole request so latency/steps/grounding land in traces.jsonl.
        with Tracer("agentic_query") as tracer:
            result = self._run(question)
            result = self._finalize(result)
            tracer.update(
                question=question,
                steps=result["steps"],
                tools_used=result["tools_used"],
                retrieved_count=len(result.get("retrieved_contexts", [])),
                grounded=result["guardrails"]["grounded"],
                grounding_score=result["guardrails"]["grounding_score"],
            )
            return result

    def _run(self, question: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        tool_calls: List[Dict[str, str]] = []
        retrieved_chunks: List[str] = []

        for step in range(self.max_steps):
            logger.info("Agent step %d/%d", step + 1, self.max_steps)
            response = chat(model=self.model_name, messages=messages)

            content = response.message.content.strip()

            tool_name, tool_args = self._parse_tool_call(content)

            if tool_name and tool_name in self.tools:
                result = self.tools[tool_name](tool_args)
                tool_calls.append({
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result[:300],
                })
                if tool_name == "search_docs":
                    retrieved_chunks.append(result)

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result ({tool_name}):\n{result}\n\n"
                        "Continue — call another tool if needed, or provide the final answer."
                    ),
                })
                continue

            # No tool call → final answer
            return {
                "answer": content,
                "steps": step + 1,
                "tool_calls": tool_calls,
                "retrieved_contexts": retrieved_chunks,
            }

        return {
            "answer": "Could not answer within the step limit. Try a simpler question.",
            "steps": self.max_steps,
            "tool_calls": tool_calls,
            "retrieved_contexts": retrieved_chunks,
        }
