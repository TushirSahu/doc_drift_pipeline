
import logging
import re
from typing import Any, Dict, List, TypedDict

from src.core import llm
from src.core.llm import chat as llm_chat
from src.core.prompts import load_prompt
from src.core.settings import cfg
from src.agentic.tools import get_enabled_tools, tool_descriptions
from src.agentic.guardrails import (
    CANARY_DIRECTIVE, check_answer, check_input, check_output,
)
from src.observability.tracing import Tracer

logger = logging.getLogger(__name__)

TOOL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*ARGS:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _clean_args(raw: str) -> str:
    """Normalize tool args the model may wrap as JSON / quotes / parens.

    Small models often emit ``["auth.md"]`` or ``("query")`` instead of a plain
    phrase. Strip the wrapping so search_docs gets a usable query.
    """
    raw = raw.strip()
    # take only the first line — models sometimes append commentary
    raw = raw.splitlines()[0].strip() if raw else raw
    for pair in ('()', '[]', '{}'):
        if raw.startswith(pair[0]) and raw.endswith(pair[1]):
            raw = raw[1:-1].strip()
    raw = raw.strip().strip('"\'').strip()
    # collapse a JSON-list-of-one like "a", "b" -> a b
    raw = raw.replace('", "', ' ').replace("', '", ' ').strip('"\'')
    return raw


def _word_chunks(text: str):
    """Yield small chunks (word + space) so a UI can 'type out' the answer."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word if i == len(words) - 1 else word + " "


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
    input_guard: Dict[str, Any]
    output_guard: Dict[str, Any]
    blocked: bool


class AgenticController:
    def __init__(self, model_name: str | None = None, spec: "llm.ModelSpec | None" = None):
        # Serving model resolution: an explicit spec/model wins; otherwise use the
        # benchmark champion if one has been recorded, else the configured default.
        # This is how "answer based on the eval scores" reaches the serving path.
        if spec is None and model_name is None:
            spec = llm.default_chat_spec()
        self.spec = spec
        self.model_name = model_name or (spec.model if spec else cfg("models", "llm", default="llama3.2:3b"))
        self.max_steps = cfg("agentic", "max_steps", default=5)
        self.tools = get_enabled_tools()

    def _system_prompt(self) -> str:
        base = load_prompt("agent_system")
        # The canary directive must be part of the prompt the model actually
        # sees, so check_output can detect it leaking back out in an answer.
        return (f"{base}\n\nAvailable tools:\n{tool_descriptions()}"
                f"\n\n{CANARY_DIRECTIVE}")

    def _blocked_message(self) -> str:
        return cfg(
            "guardrails", "blocked_message",
            default=("I can't help with that request. I answer questions about the "
                     "DocDrift documentation — please ask about the docs."),
        )

    def _parse_tool_call(self, text: str) -> tuple[str | None, str | None]:
        # 1) Strict format: "TOOL: <name> ARGS: <args>".
        match = TOOL_PATTERN.search(text)
        if match:
            return match.group(1).lower().strip(), _clean_args(match.group(2))

        # 2) Lenient: small models often drop the "TOOL:" prefix or use call
        #    syntax. Accept "<tool> ARGS: <args>" or "<tool>(<args>)" for any
        #    enabled tool, so a malformed tool call still triggers retrieval
        #    instead of leaking into the final answer.
        names = "|".join(re.escape(n) for n in self.tools)
        if names:
            lenient = re.search(
                rf"\b({names})\b\s*(?:ARGS:\s*(.+)|\((.+)\))",
                text, re.IGNORECASE | re.DOTALL,
            )
            if lenient:
                args = lenient.group(2) if lenient.group(2) is not None else lenient.group(3)
                return lenient.group(1).lower().strip(), _clean_args(args)
        return None, None

    def _is_final_answer(self, text: str) -> bool:
        """If the LLM didn't request a tool, treat the message as the final answer."""
        tool_name, _ = self._parse_tool_call(text)
        return tool_name is None

    def _finalize(self, result: AgentResult) -> AgentResult:
        """Attach guardrail verdict + tool-usage summary to a result dict."""
        result.setdefault("input_guard", {"allowed": True, "category": None,
                                           "reasons": ["passed"]})
        result.setdefault("output_guard", {"leaked": False, "category": None,
                                            "overlap": 0.0, "reasons": ["passed"]})
        # A request refused by the input filter never ran the model, so the
        # answer-grounding check doesn't apply — record a passthrough verdict.
        if result.get("blocked"):
            result["guardrails"] = {
                "grounded": True, "grounding_score": 0.0, "has_citation": False,
                "is_idk": False, "reasons": ["input blocked by guardrail"],
            }
            result["tools_used"] = []
            return result

        # Layer 3 — screen the final answer for leaked system-prompt content
        # before it leaves the controller. On a leak, withhold the answer and
        # return the safe refusal instead (the trace still records the attempt).
        if cfg("guardrails", "output_filter", default=True):
            out = check_output(result["answer"], self._system_prompt())
            result["output_guard"] = out.to_dict()
            if out.leaked:
                logger.warning("Output guardrail withheld a leak: %s", out.category)
                result["answer"] = self._blocked_message()
                result["retrieved_contexts"] = []

        verdict = check_answer(result["answer"], result.get("retrieved_contexts", []))
        result["guardrails"] = verdict.to_dict()
        result["tools_used"] = sorted({c["tool"] for c in result.get("tool_calls", [])})
        return result

    def _iter(self, question: str):
        """Core agent loop as a generator.

        Yields ``("step", call_info)`` for each tool call as it happens, then
        exactly one ``("final", result_dict)``. Both run() and run_stream()
        consume this, so the loop logic lives in one place.
        """
        # Layer 2 — screen the question for prompt injection before the model
        # ever sees it. If flagged, refuse immediately (no LLM call, no tools).
        if cfg("guardrails", "input_filter", default=True):
            guard = check_input(question)
            if not guard.allowed:
                logger.warning("Input guardrail blocked a request: %s", guard.category)
                yield "final", {
                    "answer": self._blocked_message(),
                    "steps": 0,
                    "tool_calls": [],
                    "retrieved_contexts": [],
                    "blocked": True,
                    "input_guard": guard.to_dict(),
                }
                return
            input_guard = guard.to_dict()
        else:
            input_guard = {"allowed": True, "category": None, "reasons": ["disabled"]}

        # Layer 4 — wrap the user question in a delimiter and feed it in the
        # user turn so the model treats it as untrusted DATA, not instructions.
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"<user_question>\n{question}\n</user_question>"},
        ]
        tool_calls: List[Dict[str, str]] = []
        retrieved_chunks: List[str] = []

        for step in range(self.max_steps):
            logger.info("Agent step %d/%d", step + 1, self.max_steps)
            content = llm_chat(messages, model=self.model_name, spec=self.spec).strip()
            tool_name, tool_args = self._parse_tool_call(content)

            if tool_name and tool_name in self.tools:
                result = self.tools[tool_name](tool_args)
                call = {
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result[:300],
                }
                tool_calls.append(call)
                if tool_name == "search_docs":
                    retrieved_chunks.append(result)

                yield "step", call
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result ({tool_name}):\n{result}\n\n"
                        "Continue — call another tool if needed, or provide the final answer."
                    ),
                })
                continue

            yield "final", {
                "answer": content,
                "steps": step + 1,
                "tool_calls": tool_calls,
                "retrieved_contexts": retrieved_chunks,
                "blocked": False,
                "input_guard": input_guard,
            }
            return

        yield "final", {
            "answer": "Could not answer within the step limit. Try a simpler question.",
            "steps": self.max_steps,
            "tool_calls": tool_calls,
            "retrieved_contexts": retrieved_chunks,
            "blocked": False,
            "input_guard": input_guard,
        }

    def run(self, question: str) -> AgentResult:
        # Trace the whole request so latency/steps/grounding land in traces.jsonl.
        with Tracer("agentic_query") as tracer:
            result: AgentResult = {}
            for kind, payload in self._iter(question):
                if kind == "final":
                    result = payload
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

    def run_stream(self, question: str):
        """Stream the answer as events: ``step`` (per tool) → ``token`` → ``done``.

        The final answer is sent in word chunks so the client can type it out.
        """
        final: AgentResult = {}
        for kind, payload in self._iter(question):
            if kind == "step":
                yield {"type": "step", "tool": payload["tool"], "args": payload["args"]}
            else:
                final = payload

        result = self._finalize(final)
        for chunk in _word_chunks(result["answer"]):
            yield {"type": "token", "text": chunk}
        yield {
            "type": "done",
            "steps": result["steps"],
            "tools_used": result["tools_used"],
            "retrieved_contexts": result.get("retrieved_contexts", []),
            "guardrails": result["guardrails"],
            "input_guard": result.get("input_guard", {}),
            "blocked": result.get("blocked", False),
        }
