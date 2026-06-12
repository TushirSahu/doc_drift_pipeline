from src.agentic.controller import TOOL_PATTERN, AgenticController
from src.agentic.tools import calculator, get_enabled_tools


def test_calculator_basic():
    assert calculator("2 + 3") == "5.0"
    assert calculator("12 * 60") == "720.0"


def test_calculator_rejects_unsafe_code():
    result = calculator("__import__('os').system('ls')")
    assert "error" in result.lower()


def test_tool_pattern_parses_search_call():
    text = "TOOL: search_docs ARGS: auth token expiry"
    match = TOOL_PATTERN.search(text)
    assert match
    assert match.group(1).lower() == "search_docs"
    assert "auth token" in match.group(2)


def test_enabled_tools_whitelist():
    tools = get_enabled_tools()
    assert "search_docs" in tools
    assert "calculator" in tools
    assert "web_search" not in tools  # disabled in config


def test_controller_detects_final_answer():
    controller = AgenticController()
    assert controller._is_final_answer("Auth v2 uses OAuth2 and JWT tokens.")
    assert not controller._is_final_answer("TOOL: search_docs ARGS: auth methods")
