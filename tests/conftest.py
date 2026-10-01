"""
Shared test fixtures. pytest loads this file automatically.

No test here ever calls the real OpenAI API: the client is replaced with a fake
(MagicMock) whose responses we define per test. Tests are free, fast and deterministic.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import agent
from app.config import get_settings


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Every test gets a fake API key, regardless of what is in .env or the shell."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_openai(monkeypatch):
    """Replaces get_openai_client() with a fake client and returns it to the test."""
    fake_client = MagicMock()
    monkeypatch.setattr(agent, "get_openai_client", lambda: fake_client)
    return fake_client


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
