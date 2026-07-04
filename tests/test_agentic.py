from src.agentic.controller import TOOL_PATTERN, AgenticController, _clean_args
from src.agentic.tools import calculator, get_enabled_tools


def test_calculator_basic():
    assert calculator("2 + 3") == "5.0"
    assert calculator("12 * 60") == "720.0"


def test_calculator_rejects_unsafe_code():
    result = calculator("__import__('os').system('ls')")
    assert "error" in result.lower()


def test_calculator_rejects_huge_power():
    # Resource-exhaustion guard: must not try to compute 9**9**9.
    assert "error" in calculator("9**9**9").lower()
    assert "error" in calculator("10**500").lower()
    # normal powers still work
    assert calculator("2 ** 10") == "1024.0"


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


def test_clean_args_strips_json_and_quotes():
    assert _clean_args('["auth_service_v2.md"]') == "auth_service_v2.md"
    assert _clean_args('("admin session")') == "admin session"
    assert _clean_args('"refresh token"') == "refresh token"
    assert _clean_args('plain phrase') == "plain phrase"


def test_parser_recovers_tool_call_without_prefix():
    # The real failure: model dropped the "TOOL:" prefix.
    controller = AgenticController()
    name, args = controller._parse_tool_call('search_docs ARGS: ["auth_service_v2.md"]')
    assert name == "search_docs"
    assert args == "auth_service_v2.md"  # cleaned for retrieval
    # such a message must NOT be treated as the final answer anymore
    assert not controller._is_final_answer('search_docs ARGS: ["auth_service_v2.md"]')


def test_parser_accepts_call_syntax():
    controller = AgenticController()
    name, args = controller._parse_tool_call('calculator(12 * 60)')
    assert name == "calculator"
    assert args == "12 * 60"


def test_strict_format_still_works():
    controller = AgenticController()
    name, args = controller._parse_tool_call("TOOL: search_docs ARGS: auth token expiry")
    assert name == "search_docs"
    assert "auth token" in args
