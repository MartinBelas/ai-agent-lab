"""HTTP layer tests: endpoints, request validation. The agent itself is replaced."""

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from app.main import MAX_MESSAGE_LENGTH, AgentRunner, app, get_agent

client = TestClient(app)


@pytest.fixture
def use_agent():
    """
    Replaces the agent behind /api/chat via FastAPI dependency overrides.
    Usage in a test: use_agent(lambda message: "reply").
    """

    def override(fake_agent: AgentRunner) -> None:
        app.dependency_overrides[get_agent] = lambda: fake_agent

    yield override
    app.dependency_overrides.clear()


def test_health_returns_up():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_chat_returns_agent_reply(use_agent):
    use_agent(lambda message: f"echo: {message}")

    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: Hi"}


def test_chat_rejects_missing_message():
    response = client.post("/api/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_empty_message():
    response = client.post("/api/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_too_long_message(use_agent):
    def must_not_be_called(message):
        raise AssertionError("agent must not be called for an invalid request")

    use_agent(must_not_be_called)

    response = client.post("/api/chat", json={"message": "x" * (MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 422


def test_chat_maps_openai_error_to_502(use_agent):
    def failing_agent(message):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        raise APIConnectionError(message="secret internal detail", request=request)

    use_agent(failing_agent)

    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The AI provider is currently unavailable."}
    assert "secret internal detail" not in response.text
