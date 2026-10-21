"""Request ID (X-Request-ID) tests: taken from the caller or generated, logged, returned and forwarded."""

import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from app.agent import AgentResult
from app.main import LOG_FORMAT, app, configure_logging, get_agent
from app.request_context import REQUEST_ID_HEADER, get_request_id
from tests.conftest import text_response, tool_call_response

client = TestClient(app)


@pytest.fixture
def seen_request_ids():
    """Replaces the agent with one that records the request ID it sees while running."""
    seen: list[str | None] = []

    def recording_agent(message, history):
        seen.append(get_request_id())
        return AgentResult(reply="ok")

    app.dependency_overrides[get_agent] = lambda: recording_agent
    yield seen
    app.dependency_overrides.clear()


def request_id_of(record: logging.LogRecord) -> str:
    """record.request_id, added at runtime by the log record factory (unknown to type checkers)."""
    return vars(record)["request_id"]


def chat(headers: dict[str, str] | None = None) -> httpx.Response:
    return client.post("/api/chat", json={"message": "Hi"}, headers=headers or {})


def test_request_id_from_the_caller_is_used_and_returned(seen_request_ids):
    response = chat({REQUEST_ID_HEADER: "a1b2c3d4"})

    assert response.headers[REQUEST_ID_HEADER] == "a1b2c3d4"
    # Visible inside the (sync) endpoint, which runs in a worker thread.
    assert seen_request_ids == ["a1b2c3d4"]


def test_request_id_is_generated_when_missing(seen_request_ids):
    response = chat()

    request_id = response.headers[REQUEST_ID_HEADER]
    assert re.fullmatch(r"[0-9a-f]{8}", request_id)
    assert seen_request_ids == [request_id]


def test_every_request_gets_its_own_id(seen_request_ids):
    chat()
    chat()

    assert seen_request_ids[0] != seen_request_ids[1]


@pytest.mark.parametrize(
    "invalid_id",
    ["has spaces", "semi;colon", "<script>", "x" * 65, ""],
)
def test_invalid_request_id_is_replaced_and_not_logged(seen_request_ids, caplog, invalid_id):
    caplog.set_level(logging.WARNING, logger="app.request_context")

    response = chat({REQUEST_ID_HEADER: invalid_id})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id != invalid_id
    assert re.fullmatch(r"[0-9a-f]{8}", request_id)
    assert seen_request_ids == [request_id]
    assert "Ignoring invalid X-Request-ID header" in caplog.text
    if invalid_id:
        assert invalid_id not in caplog.text
    # The warning itself can be found by the ID the caller got back.
    warnings = [record for record in caplog.records if record.getMessage().startswith("Ignoring invalid")]
    assert [request_id_of(record) for record in warnings] == [request_id]


def test_request_id_is_not_set_after_the_request(seen_request_ids):
    chat({REQUEST_ID_HEADER: "a1b2c3d4"})

    assert get_request_id() is None


def test_error_responses_carry_the_request_id_too():
    def failing_agent(message, history):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.test"))

    app.dependency_overrides[get_agent] = lambda: failing_agent
    try:
        response = chat({REQUEST_ID_HEADER: "a1b2c3d4"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.headers[REQUEST_ID_HEADER] == "a1b2c3d4"


def test_unhandled_errors_carry_the_request_id_too(caplog):
    def crashing_agent(message, history):
        raise RuntimeError("bug")

    app.dependency_overrides[get_agent] = lambda: crashing_agent
    try:
        # Like a real server: the exception becomes a 500 instead of failing the test.
        response = TestClient(app, raise_server_exceptions=False).post(
            "/api/chat", json={"message": "Hi"}, headers={REQUEST_ID_HEADER: "a1b2c3d4"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers[REQUEST_ID_HEADER] == "a1b2c3d4"
    errors = [record for record in caplog.records if record.getMessage().startswith("Unhandled RuntimeError")]
    assert [request_id_of(record) for record in errors] == ["a1b2c3d4"]
    assert errors[0].exc_info is not None  # with the traceback


def test_validation_errors_carry_the_request_id_too():
    response = client.post("/api/chat", json={"message": ""}, headers={REQUEST_ID_HEADER: "a1b2c3d4"})

    assert response.status_code == 422
    assert response.headers[REQUEST_ID_HEADER] == "a1b2c3d4"


def format_record() -> str:
    """Creates a record the way every logger does and formats it with the app's format."""
    record = logging.getLogRecordFactory()(
        "httpx", logging.INFO, __file__, 1, "HTTP Request: POST ...", None, None
    )
    return logging.Formatter(LOG_FORMAT).format(record)


def test_log_lines_outside_a_request_show_a_dash():
    assert format_record() == "INFO:     [-] httpx - HTTP Request: POST ..."


def test_log_lines_during_a_request_show_its_id(seen_request_ids):
    formatted: list[str] = []

    def logging_agent(message, history):
        formatted.append(format_record())
        return AgentResult(reply="ok")

    app.dependency_overrides[get_agent] = lambda: logging_agent
    chat({REQUEST_ID_HEADER: "a1b2c3d4"})

    assert formatted == ["INFO:     [a1b2c3d4] httpx - HTTP Request: POST ..."]


def test_request_id_is_on_records_when_logging_is_configured_by_someone_else(seen_request_ids, caplog):
    # pytest has already put its own handler on the root logger, like uvicorn --log-config
    # or a hosting platform would; configure_logging() must not depend on adding its own.
    assert logging.getLogger().handlers
    caplog.set_level(logging.INFO, logger="app.main")

    chat({REQUEST_ID_HEADER: "a1b2c3d4"})  # the real endpoint logs "Chat request ..."

    chat_lines = [record for record in caplog.records if record.getMessage().startswith("Chat request")]
    assert [request_id_of(record) for record in chat_lines] == ["a1b2c3d4"]


def test_records_outside_a_request_have_a_dash(caplog):
    caplog.set_level(logging.INFO, logger="app.test")

    logging.getLogger("app.test").info("startup")

    assert request_id_of(caplog.records[-1]) == "-"


def test_configure_logging_can_be_called_again():
    factory = logging.getLogRecordFactory()

    configure_logging()

    assert logging.getLogRecordFactory() is factory


def test_request_id_is_forwarded_to_ai_demo(fake_openai, fake_ai_demo):
    sent = fake_ai_demo(lambda request: httpx.Response(200, json={"providers": []}))
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_ai_demo_providers"),
        text_response("No providers right now."),
    ]

    response = chat({REQUEST_ID_HEADER: "a1b2c3d4"})

    assert response.status_code == 200
    assert sent[0].headers[REQUEST_ID_HEADER] == "a1b2c3d4"


def test_no_request_id_is_sent_to_ai_demo_outside_a_request(fake_ai_demo):
    from app import ai_demo

    sent = fake_ai_demo(lambda request: httpx.Response(200, json={"providers": []}))

    ai_demo.get_ai_demo_providers()

    assert REQUEST_ID_HEADER not in sent[0].headers


def test_request_id_is_not_sent_to_tavily(fake_openai, fake_tavily):
    sent = fake_tavily(lambda request: httpx.Response(200, json={"results": []}))
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", '{"query": "EUR CZK"}'),
        text_response("No results."),
    ]

    chat({REQUEST_ID_HEADER: "a1b2c3d4"})

    assert REQUEST_ID_HEADER not in sent[0].headers
