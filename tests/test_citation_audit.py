"""Citation-accuracy audit (Layer: verify + auto-correct [Source: ...])."""
from src.agentic.controller import AgenticController
from src.agentic.guardrails import (
    _contexts_by_source, check_citation_accuracy, rewrite_citation,
)

# Source-tagged contexts, exactly as search_docs now formats them.
AUTH_CTX = "[Chunk 1 | Source: auth.md]\nOAuth2 and JWT tokens secure the service."
BILLING_CTX = "[Chunk 1 | Source: billing.md]\nRefunds processed by finance team."


def test_accurate_citation():
    ans = "OAuth2 and JWT tokens secure the service. [Source: auth.md]"
    r = check_citation_accuracy(ans, [AUTH_CTX])
    assert r.verdict == "accurate"
    assert r.corrected_to is None


def test_corrected_citation_points_to_the_real_source():
    # Answer is grounded in auth.md but cites billing.md.
    ans = "OAuth2 and JWT tokens secure the service. [Source: billing.md]"
    r = check_citation_accuracy(ans, [BILLING_CTX, AUTH_CTX])
    assert r.verdict == "corrected"
    assert r.corrected_to == "auth.md"


def test_invented_citation():
    ans = "OAuth2 and JWT tokens secure the service. [Source: ghost.md]"
    r = check_citation_accuracy(ans, [BILLING_CTX])  # ghost.md never retrieved
    assert r.verdict == "invented"


def test_unsupported_citation():
    # Cited source is retrieved, but the answer shares nothing with it.
    ans = "Biometric retina scanning protects everything. [Source: auth.md]"
    r = check_citation_accuracy(ans, [AUTH_CTX])
    assert r.verdict == "unsupported"
    assert r.corrected_to is None


def test_no_citation_is_skipped():
    r = check_citation_accuracy("OAuth2 secures the service.", [AUTH_CTX])
    assert r.verdict == "no_citation"


def test_rewrite_citation_replaces_source():
    ans = "OAuth2 secures it. [Source: wrong.md]"
    assert rewrite_citation(ans, "auth.md") == "OAuth2 secures it. [Source: auth.md]"


def test_contexts_by_source_splits_multiple_chunks():
    ctx = ("[Chunk 1 | Source: a.md]\nalpha text\n\n"
           "[Chunk 2 | Source: b.md]\nbeta text")
    by = _contexts_by_source([ctx])
    assert "alpha text" in by["a.md"][0]
    assert "beta text" in by["b.md"][0]


def test_controller_autocorrects_miscited_answer():
    c = AgenticController()
    result = c._finalize({
        "answer": "OAuth2 and JWT tokens secure the service. [Source: billing.md]",
        "retrieved_contexts": [BILLING_CTX, AUTH_CTX],
        "blocked": False,
        "tool_calls": [],
    })
    assert result["citation_audit"]["verdict"] == "corrected"
    assert "[Source: auth.md]" in result["answer"]
    assert "[Source: billing.md]" not in result["answer"]
