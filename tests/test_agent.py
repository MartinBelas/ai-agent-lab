"""Agent loop tests with a fake OpenAI client (see conftest.py)."""

import time
from types import SimpleNamespace

import httpx

from app import agent
from app.config import get_settings
from app.prompts import STEP_LIMIT_PROMPT
from app.tool_result import Source, ToolResult
from tests.conftest import tavily_results, text_response, tool_call_response


def test_tool_schemas_match_handlers():
    """Every tool the model is told about must have a handler, and vice versa."""
    schema_names = {schema["function"]["name"] for schema in agent.TOOL_SCHEMAS}

    assert schema_names == set(agent.TOOL_HANDLERS)


def test_answer_without_tool(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("Hello!")]

    reply = agent.run_agent("Hi").reply

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

    reply = agent.run_agent("What time is it?").reply

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

    reply = agent.run_agent("Delete everything").reply

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

    reply = agent.run_agent("What time is it?").reply

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

    reply = agent.run_agent("What time is it?").reply

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

    reply = agent.run_agent("What time is it?").reply

    assert reply == "Something went wrong."
    tool_message = fake_openai.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert "not valid JSON" in tool_message["content"]


def test_unexpected_arguments_are_reported_to_model(fake_openai):
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_current_time", arguments='{"timezone": "CET"}'),
        text_response("Here is the UTC time instead."),
    ]

    reply = agent.run_agent("What time is it in Prague?").reply

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

    reply = agent.run_agent("Check the time twice").reply

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

    reply = agent.run_agent("Loop forever").reply

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

    reply = agent.run_agent("Check the time twice").reply

    assert reply == "Done."
    assert len(executions) == 1
    third_call_messages = fake_openai.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert third_call_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": "Error: tool call limit for this request reached.",
    }


# --- web_search in the agent loop ---------------------------------------

def offered_tool_names(create_call) -> set[str]:
    return {schema["function"]["name"] for schema in create_call.kwargs["tools"]}


def test_web_search_is_offered_when_key_is_configured(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("Hi")]

    agent.run_agent("Hi")

    first_call = fake_openai.chat.completions.create.call_args_list[0]
    assert "web_search" in offered_tool_names(first_call)


def test_web_search_is_not_offered_without_key(fake_openai, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "news"}'),
        text_response("I can't search the web."),
    ]

    agent.run_agent("What's new today?")

    calls = fake_openai.chat.completions.create.call_args_list
    assert offered_tool_names(calls[0]) == {"get_current_time", "get_ai_demo_providers"}
    # Even if the model asks for it anyway, the tool is not run.
    assert calls[1].kwargs["messages"][-1]["content"] == "Unknown tool: web_search"


def test_web_search_has_its_own_per_request_limit(fake_openai, monkeypatch):
    searches = []
    monkeypatch.setitem(agent.TOOL_HANDLERS, "web_search", lambda query: searches.append(query) or "results")
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    monkeypatch.setenv("WEB_SEARCH_MAX_CALLS", "2")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "a"}', call_id="call_1"),
        tool_call_response("web_search", arguments='{"query": "b"}', call_id="call_2"),
        tool_call_response("web_search", arguments='{"query": "c"}', call_id="call_3"),
        tool_call_response("get_current_time", call_id="call_4"),
        text_response("Done."),
    ]

    reply = agent.run_agent("Search a lot").reply

    assert reply == "Done."
    assert searches == ["a", "b"]
    # The mock keeps a reference to the same (growing) messages list for every call,
    # so look the tool results up by their call id in the final conversation.
    final_messages = fake_openai.chat.completions.create.call_args_list[-1].kwargs["messages"]
    tool_results = {m["tool_call_id"]: m["content"] for m in final_messages if m["role"] == "tool"}
    assert tool_results["call_1"] == "results"
    assert "at most 2 times per request" in tool_results["call_3"]
    # Other tools are not affected by the web_search limit.
    assert tool_results["call_4"] == "12:00"


def test_web_search_results_go_back_to_the_model(fake_openai, fake_tavily):
    fake_tavily(lambda request: httpx.Response(200, json=tavily_results(
        {"title": "Release notes", "url": "https://example.com/notes", "content": "Version 2.0 is out."}
    )))
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "latest version"}', call_id="call_1"),
        text_response("Version 2.0 (https://example.com/notes)."),
    ]

    reply = agent.run_agent("What is the latest version?").reply

    assert reply == "Version 2.0 (https://example.com/notes)."
    tool_message = fake_openai.chat.completions.create.call_args_list[-1].kwargs["messages"][-1]
    assert tool_message["tool_call_id"] == "call_1"
    assert "URL: https://example.com/notes" in tool_message["content"]


# --- Conversation history ---------------------------------------------

def test_history_is_sent_between_system_prompt_and_new_message(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("About 28 USD.")]
    history = [
        agent.HistoryMessage(role="user", content="How much is 600 CZK in EUR?"),
        agent.HistoryMessage(role="assistant", content="About 24.70 EUR."),
    ]

    agent.run_agent("And in USD?", history)

    messages = fake_openai.chat.completions.create.call_args_list[0].kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:] == [
        {"role": "user", "content": "How much is 600 CZK in EUR?"},
        {"role": "assistant", "content": "About 24.70 EUR."},
        {"role": "user", "content": "And in USD?"},
    ]


def test_without_history_only_system_prompt_and_message_are_sent(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("Hi")]

    agent.run_agent("Hi")

    messages = fake_openai.chat.completions.create.call_args_list[0].kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


# --- Sources ------------------------------------------------------------

def search_result(*sources: tuple[str, str]) -> ToolResult:
    return ToolResult(content="results", sources=tuple(Source(title=t, url=u) for t, u in sources))


def test_sources_from_all_searches_are_collected_without_duplicates(fake_openai, monkeypatch):
    results = iter([
        search_result(("A", "https://a.example"), ("B", "https://b.example")),
        search_result(("B again", "https://b.example"), ("C", "https://c.example")),
    ])
    monkeypatch.setitem(agent.TOOL_HANDLERS, "web_search", lambda query: next(results))
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "one"}', call_id="call_1"),
        tool_call_response("web_search", arguments='{"query": "two"}', call_id="call_2"),
        text_response("Answer."),
    ]

    result = agent.run_agent("Search twice")

    assert result.sources == (
        Source(title="A", url="https://a.example"),
        Source(title="B", url="https://b.example"),  # first occurrence wins
        Source(title="C", url="https://c.example"),
    )


def test_only_tool_content_goes_to_the_model(fake_openai, monkeypatch):
    monkeypatch.setitem(
        agent.TOOL_HANDLERS, "web_search", lambda query: search_result(("A", "https://a.example"))
    )
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "x"}', call_id="call_1"),
        text_response("Answer."),
    ]

    agent.run_agent("Search")

    tool_message = fake_openai.chat.completions.create.call_args_list[-1].kwargs["messages"][-1]
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": "results"}


def test_answer_without_search_has_no_sources(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("4")]

    assert agent.run_agent("2+2?").sources == ()


def test_refused_search_adds_no_sources(fake_openai, monkeypatch):
    monkeypatch.setitem(
        agent.TOOL_HANDLERS, "web_search", lambda query: search_result(("A", "https://a.example"))
    )
    monkeypatch.setenv("WEB_SEARCH_MAX_CALLS", "1")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("web_search", arguments='{"query": "x"}', call_id="call_1"),
        tool_call_response("web_search", arguments='{"query": "y"}', call_id="call_2"),
        text_response("Answer."),
    ]

    result = agent.run_agent("Search a lot")

    assert result.sources == (Source(title="A", url="https://a.example"),)
    assert result.usage.web_searches == 1


# --- Usage ------------------------------------------------------------

def with_tokens(response, prompt_tokens: int, completion_tokens: int):
    response.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def test_usage_counts_model_calls_tools_and_tokens(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    monkeypatch.setitem(agent.TOOL_HANDLERS, "web_search", lambda query: "results")
    fake_openai.chat.completions.create.side_effect = [
        with_tokens(tool_call_response("web_search", arguments='{"query": "x"}', call_id="c1"), 100, 10),
        with_tokens(tool_call_response("get_current_time", call_id="c2"), 300, 5),
        with_tokens(text_response("Done."), 400, 50),
    ]

    usage = agent.run_agent("Do things").usage

    assert usage == agent.AgentUsage(
        model_calls=3, tool_calls=2, web_searches=1, input_tokens=800, output_tokens=65
    )


def test_usage_includes_the_forced_final_call(fake_openai, monkeypatch):
    monkeypatch.setitem(agent.TOOL_HANDLERS, "get_current_time", lambda: "12:00")
    monkeypatch.setenv("AGENT_MAX_STEPS", "1")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        with_tokens(tool_call_response("get_current_time"), 100, 10),
        with_tokens(text_response("Final."), 200, 20),
    ]

    usage = agent.run_agent("Loop").usage

    assert usage.model_calls == 2
    assert (usage.input_tokens, usage.output_tokens) == (300, 30)


def test_unknown_tool_is_not_counted_as_tool_call(fake_openai):
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("delete_all_files"),
        text_response("No."),
    ]

    assert agent.run_agent("Delete everything").usage.tool_calls == 0


# --- get_ai_demo_providers in the agent loop ------------------------------

def test_ai_demo_providers_is_offered_when_url_is_configured(fake_openai):
    fake_openai.chat.completions.create.side_effect = [text_response("Hi")]

    agent.run_agent("Hi")

    first_call = fake_openai.chat.completions.create.call_args_list[0]
    assert "get_ai_demo_providers" in offered_tool_names(first_call)


def test_ai_demo_providers_is_not_offered_without_url(fake_openai, monkeypatch):
    monkeypatch.setenv("AI_DEMO_BASE_URL", "")
    get_settings.cache_clear()
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_ai_demo_providers"),
        text_response("I can't check that."),
    ]

    agent.run_agent("Which models does this demo support?")

    calls = fake_openai.chat.completions.create.call_args_list
    assert "get_ai_demo_providers" not in offered_tool_names(calls[0])
    assert calls[1].kwargs["messages"][-1]["content"] == "Unknown tool: get_ai_demo_providers"


def test_ai_demo_providers_result_goes_back_to_the_model(fake_openai, fake_ai_demo):
    fake_ai_demo(lambda request: httpx.Response(200, json={"providers": [{"id": "OPENAI", "model": "gpt-x"}]}))
    fake_openai.chat.completions.create.side_effect = [
        tool_call_response("get_ai_demo_providers", call_id="call_1"),
        text_response("This demo supports OpenAI (gpt-x)."),
    ]

    result = agent.run_agent("Which models does this demo support?")

    assert result.reply == "This demo supports OpenAI (gpt-x)."
    assert result.sources == ()  # not a web search
    assert result.usage.tool_calls == 1
    assert result.usage.web_searches == 0
    tool_message = fake_openai.chat.completions.create.call_args_list[-1].kwargs["messages"][-1]
    assert tool_message["tool_call_id"] == "call_1"
    assert "- OPENAI (model: gpt-x)" in tool_message["content"]
