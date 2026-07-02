"""
Naive RAG:  question → retrieve once → answer
Agentic RAG: question → LLM decides → maybe search again → maybe use tools → answer

The LLM is the *controller*; retrieval is just one tool it can call.
"""
from src.agentic.controller import AgenticController
from src.agentic.tools import TOOL_REGISTRY, get_enabled_tools

__all__ = ["AgenticController", "TOOL_REGISTRY", "get_enabled_tools"]
