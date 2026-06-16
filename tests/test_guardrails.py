from src.agentic.guardrails import check_answer, grounding_score


CONTEXT = [
    "The Auth Service v2.0 uses OAuth2 and JWT tokens. Tokens expire after 15 "
    "minutes. Admin users have a 12 hour maximum session.",
]


def test_grounded_answer_scores_high():
    answer = "Auth v2 uses OAuth2 and JWT tokens that expire after 15 minutes."
    assert grounding_score(answer, CONTEXT) > 0.6


def test_hallucinated_answer_scores_low():
    answer = "The service relies on biometric retina scanning and quantum keys."
    assert grounding_score(answer, CONTEXT) < 0.3


def test_check_flags_missing_citation():
    answer = "Auth v2 uses OAuth2 and JWT tokens, expiring after 15 minutes."
    result = check_answer(answer, CONTEXT, require_citation=True)
    assert not result.has_citation
    assert any("citation" in r for r in result.reasons)


def test_check_passes_with_citation_and_grounding():
    answer = ("Auth v2 uses OAuth2 and JWT tokens that expire after 15 minutes. "
              "[Source: auth_service_v2.md]")
    result = check_answer(answer, CONTEXT, require_citation=True)
    assert result.grounded
    assert result.has_citation


def test_check_flags_ungrounded_answer():
    answer = "It uses biometric retina scanning. [Source: made_up.md]"
    result = check_answer(answer, CONTEXT)
    assert not result.grounded
    assert any("grounding" in r for r in result.reasons)


def test_idk_answer_is_accepted():
    result = check_answer("I don't know based on the documentation.", CONTEXT)
    assert result.is_idk
    assert result.grounded


def test_no_context_is_flagged():
    result = check_answer("Auth uses OAuth2. [Source: x.md]", [])
    assert not result.grounded
