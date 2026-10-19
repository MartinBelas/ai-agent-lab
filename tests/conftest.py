"""
Shared test fixtures. pytest loads this file automatically.

No test here ever calls a real paid API: the OpenAI client is replaced with a fake
(MagicMock) whose responses we define per test, and the HTTP clients used by
web_search and get_ai_demo_providers are replaced with an httpx.MockTransport. Tests are free, fast and
deterministic.
"""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app import agent, ai_demo, web_search
from app.config import get_settings


@pytest.fixture(autouse=True)
def test_settings(monkeypatch, tmp_path):
    """Every test gets fake keys, regardless of what is in .env or the shell."""
    # Settings read .env from the working directory. Running each test in an empty
    # temporary directory makes sure the developer's real .env never leaks in.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key-not-real")
    monkeypatch.setenv("AI_DEMO_BASE_URL", "https://ai-demo.test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    """Safety net: a test that makes an HTTP call without fake_tavily / fake_ai_demo fails loudly."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Real HTTP call in a test: {request.method} {request.url}")

    def refusing_client() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(refuse))

    monkeypatch.setattr(web_search, "get_http_client", refusing_client)
    monkeypatch.setattr(ai_demo, "get_http_client", refusing_client)


@pytest.fixture
def fake_openai(monkeypatch):
    """Replaces get_openai_client() with a fake client and returns it to the test."""
    fake_client = MagicMock()
    monkeypatch.setattr(agent, "get_openai_client", lambda: fake_client)
    return fake_client


def fake_http_api(monkeypatch, module):
    """
    Returns an installer that lets a test decide how an external HTTP API answers.
    Call it with a handler that receives the httpx.Request and returns an
    httpx.Response (or raises). The requests that were sent are collected in the
    returned list.
    """
    sent_requests: list[httpx.Request] = []

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list[httpx.Request]:
        def recording_handler(request: httpx.Request) -> httpx.Response:
            sent_requests.append(request)
            return handler(request)

        client = httpx.Client(transport=httpx.MockTransport(recording_handler))
        monkeypatch.setattr(module, "get_http_client", lambda: client)
        return sent_requests

    return install


@pytest.fixture
def fake_tavily(monkeypatch):
    """The Tavily Search API, as used by web_search."""
    return fake_http_api(monkeypatch, web_search)


@pytest.fixture
def fake_ai_demo(monkeypatch):
    """The Java AI-Demo app, as used by get_ai_demo_providers."""
    return fake_http_api(monkeypatch, ai_demo)


# --- Helpers that build objects shaped like OpenAI responses ----------

def text_response(text: str):
    """The model answers with plain text, no tool call."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call_response(tool_name: str, arguments: str = "{}", call_id: str = "call_1"):
    """The model asks us to call one tool."""
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=tool_name, arguments=arguments),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tavily_results(*results: dict) -> dict:
    """A Tavily /search response body with the given results."""
    return {"query": "test", "results": list(results), "response_time": 0.1}
