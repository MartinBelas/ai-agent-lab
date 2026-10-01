"""Agent loop tests with a fake OpenAI client (see conftest.py)."""

import pytest

from app import agent
from tests.conftest import text_response, tool_call_response


def test_tool_schemas_match_handlers():
    """Every tool the model is told about must have a handler, and vice versa."""
    schema_names = {schema["function"]["name"] for schema in agent.TOOL_SCHEMAS}

    assert schema_names == set(agent.TOOL_HANDLERS)


def test_answer_without_tool(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("Hello!")]

    reply = agent.run_agent("Hi")

    assert reply == "Hello!"
    assert fake_openai.chat.completions.create.call_count == 1
    first_call = fake_openai.chat.completions.create.call_args_list[0]
    assert first_call.kwargs["model"] == "test-model"
    assert first_call.kwargs["tools"] == agent.TOOL_SCHEMAS


def test_answer_with_tool(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "2026-10-01T12:00:00+00:00")
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_42"),
        text_response("It is 12:00 UTC."),
    ]

    reply = agent.run_agent("What time is it?")

    assert reply == "It is 12:00 UTC."
    assert fake_openai.chat.completions.create.call_count == 2
    # The tool result must go back to the model as a "tool" message linked to the call id
    second_call_messages = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_message = second_call_messages[-1]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call_42",
        "content": "2026-10-01T12:00:00+00:00",
    }


def test_unknown_tool_is_reported_to_model(fake_openai):
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("delete_all_files"),
        text_response("Sorry, I can't do that."),
    ]

    reply = agent.run_agent("Delete everything")

    assert reply == "Sorry, I can't do that."
    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert tool_message["content"] == "Unknown tool: delete_all_files"


@pytest.mark.xfail(strict=True, reason="Known gap: a failing tool crashes the whole request. Fix in the next step.")
def test_failing_tool_is_reported_to_model(fake_openai, monkeypatch):
    def broken_tool():
        raise RuntimeError("tool is down")

    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", broken_tool)
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time"),
        text_response("I couldn't get the time right now."),
    ]

    reply = agent.run_agent("What time is it?")

    assert reply == "I couldn't get the time right now."
    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert "error" in tool_message["content"].lower()
