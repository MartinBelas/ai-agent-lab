"""Agent loop tests with a fake OpenAI client (see conftest.py)."""

import time

from app import agent
from app.config import get_settings
from app.prompts import STEP_LIMIT_PROMPT
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
    # Only the exception type reaches the model, not the internal message
    assert "tool is down" not in tool_message["content"]


def test_assistant_tool_call_is_sent_back_before_tool_result(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_42"),
        text_response("12:00."),
    ]

    agent.run_agent("What time is it?")

    messages = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"]
    # The API requires the assistant message with tool_calls right before the tool results
    assert messages[-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_42",
                "type": "function",
                "function": {"name": "get_current_time", "arguments": "{}"},
            }
        ],
    }
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_42"


def test_type_error_inside_tool_is_not_reported_as_invalid_arguments(fake_openai, monkeypatch):
    def buggy_tool():
        raise TypeError("bug inside the tool")  # not a problem with the model's arguments

    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", buggy_tool)
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time"),
        text_response("Sorry."),
    ]

    agent.run_agent("What time is it?")

    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert tool_message["content"] == "Error: tool 'get_current_time' failed (TypeError)."


def test_deadline_forces_final_answer_without_tools(fake_openai, monkeypatch):
    def slow_tool():
        time.sleep(0.1)
        return "12:00"

    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", slow_tool)
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_1"),
        text_response("Out of time, here is what I have."),
    ]

    reply = agent.run_agent("What time is it?")

    assert reply == "Out of time, here is what I have."
    calls = fake_openai.chat.completions.create.call_args_list
    # One round with tools, then the deadline stops the loop -> final call without tools
    assert len(calls) == 2
    assert "tools" not in calls[1].kwargs


def test_invalid_json_arguments_are_reported_to_model(fake_openai):
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", arguments="{not json"),
        text_response("Something went wrong."),
    ]

    reply = agent.run_agent("What time is it?")

    assert reply == "Something went wrong."
    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert "not valid JSON" in tool_message["content"]


def test_unexpected_arguments_are_reported_to_model(fake_openai):
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", arguments='{"timezone": "CET"}'),
        text_response("Here is the UTC time instead."),
    ]

    reply = agent.run_agent("What time is it in Prague?")

    assert reply == "Here is the UTC time instead."
    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert tool_message["content"] == "Error: invalid arguments for tool 'get_current_time'."


def test_model_can_use_tools_in_several_rounds(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_1"),
        tool_call_response("get_current_time", call_id="call_2"),
        text_response("Still 12:00."),
    ]

    reply = agent.run_agent("Check the time twice")

    assert reply == "Still 12:00."
    calls = fake_openai.chat.completions.create.call_args_list
    assert len(calls) == 3
    # Tools stay available in every round, not just the first one
    assert all(call.kwargs["tools"] == agent.TOOL_SCHEMAS for call in calls)


def test_step_limit_forces_final_answer_without_tools(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_1"),
        tool_call_response("get_current_time", call_id="call_2"),
        text_response("Final answer."),
    ]

    reply = agent.run_agent("Loop forever")

    assert reply == "Final answer."
    calls = fake_openai.chat.completions.create.call_args_list
    assert len(calls) == 3
    final_call = calls[-1]
    assert "tools" not in final_call.kwargs
    assert final_call.kwargs["messages"][-1] == {"role": "system", "content": STEP_LIMIT_PROMPT}


def test_tool_call_limit_stops_executing_tools(fake_openai, monkeypatch):
    executions = []
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: executions.append(1) or "12:00")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "1")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", call_id="call_1"),
        tool_call_response("get_current_time", call_id="call_2"),
        text_response("Done."),
    ]

    reply = agent.run_agent("Check the time twice")

    assert reply == "Done."
    assert len(executions) == 1
    third_call_messages = fake_openai.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert third_call_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": "Error: tool call limit for this request reached.",
    }
