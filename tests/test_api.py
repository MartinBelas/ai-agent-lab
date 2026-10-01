"""HTTP layer tests: endpoints, request validation. The agent itself is replaced."""

from fastapi.testclient import TestClient

from app import main
from app.main import MAX_MESSAGE_LENGTH, app

client = TestClient(app)


def test_health_returns_up():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_chat_returns_agent_reply(monkeypatch):
    monkeypatch.setattr(main, "run_agent", lambda message: f"echo: {message}")

    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json() == {"reply": "echo: Hi"}


def test_chat_rejects_missing_message():
    response = client.post("/api/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_empty_message():
    response = client.post("/api/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_too_long_message(monkeypatch):
    def must_not_be_called(message):
        raise AssertionError("agent must not be called for an invalid request")

    monkeypatch.setattr(main, "run_agent", must_not_be_called)

    response = client.post("/api/chat", json={"message": "x" * (MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 422
