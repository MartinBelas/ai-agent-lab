"""web_search tool tests. The Tavily API is faked with httpx.MockTransport (see conftest.py)."""

import json

import httpx
import pytest

from app import web_search
from app.agent import execute_tool
from app.config import get_settings
from tests.conftest import tavily_results

RESULT = {
    "title": "EUR/CZK exchange rate",
    "url": "https://example.com/eur-czk",
    "content": "1 EUR = 24.30 CZK as of today.",
    "score": 0.9,
    "published_date": "2026-10-05",
}


def test_sends_expected_request(fake_tavily):
    sent = fake_tavily(lambda request: httpx.Response(200, json=tavily_results(RESULT)))

    web_search.web_search("  EUR CZK\n rate  ")

    assert len(sent) == 1
    request = sent[0]
    assert request.method == "POST"
    assert str(request.url) == web_search.TAVILY_SEARCH_URL
    assert request.headers["Authorization"] == "Bearer tvly-test-key-not-real"
    body = json.loads(request.content)
    assert body["query"] == "EUR CZK rate"  # whitespace collapsed
    assert body["search_depth"] == "basic"  # 1 credit, not 2
    assert body["max_results"] == 3


def test_formats_results_for_the_model(fake_tavily):
    fake_tavily(lambda request: httpx.Response(200, json=tavily_results(RESULT)))

    output = web_search.web_search("EUR CZK rate")

    assert output.startswith(web_search.RESULTS_HEADER)
    assert "[1] EUR/CZK exchange rate" in output
    assert "URL: https://example.com/eur-czk" in output
    assert "Published: 2026-10-05" in output
    assert "1 EUR = 24.30 CZK as of today." in output


def test_long_excerpts_are_truncated(fake_tavily):
    long_result = {**RESULT, "content": "x" * 5000}
    fake_tavily(lambda request: httpx.Response(200, json=tavily_results(long_result)))

    output = web_search.web_search("anything")

    assert "x" * web_search.MAX_CONTENT_CHARS + "..." in output
    assert "x" * (web_search.MAX_CONTENT_CHARS + 1) not in output


def test_no_results(fake_tavily):
    fake_tavily(lambda request: httpx.Response(200, json=tavily_results()))

    assert web_search.web_search("asdfghjkl") == "No results found."


def test_empty_query_is_rejected_without_calling_tavily(fake_tavily):
    sent = fake_tavily(lambda request: httpx.Response(200, json=tavily_results(RESULT)))

    output = web_search.web_search("   ")

    assert output == "Error: the search query must not be empty."
    assert sent == []


@pytest.mark.parametrize("status_code", [429, 432, 433])
def test_quota_errors_tell_the_model_to_answer_without_live_data(fake_tavily, status_code):
    fake_tavily(lambda request: httpx.Response(status_code, json={"detail": {"error": "limit"}}))

    output = web_search.web_search("news today")

    assert "usage limit reached" in output
    assert "without live data" in output


def test_timeout_is_reported_to_the_model(fake_tavily):
    def time_out(request):
        raise httpx.ReadTimeout("too slow", request=request)

    fake_tavily(time_out)

    assert "timed out" in web_search.web_search("news today")


def test_unexpected_http_error_reaches_model_only_as_error_type(fake_tavily):
    fake_tavily(lambda request: httpx.Response(401, json={"detail": {"error": "Unauthorized: bad key"}}))

    output = execute_tool("web_search", '{"query": "news today"}')

    assert output == "Error: tool 'web_search' failed (HTTPStatusError)."


def test_missing_key_is_reported(fake_tavily, monkeypatch):
    sent = fake_tavily(lambda request: httpx.Response(200, json=tavily_results(RESULT)))
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()

    assert web_search.web_search("news today") == "Error: web search is not configured."
    assert sent == []
