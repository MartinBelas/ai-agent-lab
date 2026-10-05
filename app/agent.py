"""
The simplest possible "agent": an OpenAI model with custom tools
(tool calling / function calling). The model decides on its own whether
to call a tool, based on its description in the schema.

The agent loop:
1. Send the conversation (plus the tool schemas) to the model.
2. If the model answers with text, we are done.
3. If it asks for tools, run them, append the results and go back to 1.

The loop is bounded (agent_max_steps, agent_max_tool_calls and agent_timeout_seconds
in app/config.py), so a confused model can never spin forever or burn an unbounded
number of paid tool calls.

Tools:
- get_current_time -- a trivial sample tool
- web_search -- searches the web via Tavily (app/web_search.py); offered only when
  TAVILY_API_KEY is configured, and limited to web_search_max_calls per request

Next step: an MCP server (exposing the tools to another client, not just this agent).
"""

import inspect
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function as FunctionCallParam

from app.config import Settings, get_settings
from app.prompts import STEP_LIMIT_PROMPT, SYSTEM_PROMPT
from app.web_search import web_search

logger = logging.getLogger(__name__)


@lru_cache
def get_openai_client() -> OpenAI:
    """
    Creates the OpenAI client lazily, on the first request -- not at import time.
    Thanks to that, the module can be imported without an API key (e.g. in tests),
    and tests can replace this function with a fake client.
    """
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


# --- Definition of the tools the model can call ----------------------

def get_current_time() -> str:
    """Returns the current server time (UTC), just as a sample tool."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# What the model sees. "name" is the identifier the model sends back when it
# wants the tool called -- it must match the key in TOOL_HANDLERS.
TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches the web and returns the top results with title, URL and a short excerpt. "
                "Use it only when the answer needs current or recent information (news, prices, "
                "exchange rates, recent releases, events after your training data) or a specific "
                "fact you are not sure about. Do not use it for general knowledge, math or small talk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short, specific search query, e.g. 'EUR to CZK exchange rate today'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# Translates the tool name returned by the model to the Python function.
# Also, an allowlist: only functions listed here can ever be executed.
TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "get_current_time": get_current_time,
    "web_search": web_search,
}


def get_tool_schemas(settings: Settings) -> list[ChatCompletionToolParam]:
    """
    The tools offered to the model in this request. A tool that is not configured
    (web_search without an API key) is left out, so the model never even tries it.
    """
    return [
        schema
        for schema in TOOL_SCHEMAS
        if schema["function"]["name"] != "web_search" or settings.tavily_api_key is not None
    ]


# --- Tool budget -----------------------------------------------------

@dataclass
class ToolBudget:
    """
    Counts tool executions within one request and says when a limit is reached.
    There is an overall limit for all tools and optional stricter per-tool limits
    for tools that cost money (web_search).
    """

    max_total: int
    max_per_tool: dict[str, int]
    used_total: int = 0
    used_per_tool: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ToolBudget":
        return cls(
            max_total=settings.agent_max_tool_calls,
            max_per_tool={"web_search": settings.web_search_max_calls},
        )

    def refusal(self, tool_name: str) -> str | None:
        """None if the tool may run now, otherwise the error message for the model."""
        if self.used_total >= self.max_total:
            return "Error: tool call limit for this request reached."
        tool_limit = self.max_per_tool.get(tool_name)
        if tool_limit is not None and self.used_per_tool[tool_name] >= tool_limit:
            return (
                f"Error: tool '{tool_name}' can be used at most {tool_limit} times per request. "
                "Answer with the information you already have."
            )
        return None

    def record(self, tool_name: str) -> None:
        self.used_total += 1
        self.used_per_tool[tool_name] += 1


# --- Tool execution ------------------------------------------------

def execute_tool(tool_name: str, raw_arguments: str | None) -> str:
    """
    Runs one tool requested by the model and always returns a string.

    Any failure (unknown tool, malformed arguments, exception inside the tool)
    is turned into an error message for the model instead of crashing the
    request. Only the exception type is shown to the model -- the full details
    go to the log, so internal error messages never leak into the answer.
    """
    tool_function = TOOL_HANDLERS.get(tool_name)
    if tool_function is None:
        return f"Unknown tool: {tool_name}"

    try:
        tool_args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: arguments for tool '{tool_name}' are not valid JSON."
    if not isinstance(tool_args, dict):
        return f"Error: arguments for tool '{tool_name}' must be a JSON object."

    # Check the arguments up front, so a TypeError raised *inside* the tool (a bug)
    # is not mistaken for "the model sent wrong arguments".
    try:
        inspect.signature(tool_function).bind(**tool_args)
    except TypeError:
        logger.warning("Tool %s called with invalid arguments: %s", tool_name, tool_args)
        return f"Error: invalid arguments for tool '{tool_name}'."

    try:
        return str(tool_function(**tool_args))
    except Exception as error:
        logger.exception("Tool %s failed", tool_name)
        return f"Error: tool '{tool_name}' failed ({type(error).__name__})."


def to_assistant_message_param(message: ChatCompletionMessage) -> ChatCompletionAssistantMessageParam:
    """
    Converts the model's response message into the request format, so it can be
    sent back to the model as part of the conversation history.
    """
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=message.content,
        tool_calls=[
            ChatCompletionMessageFunctionToolCallParam(
                id=tool_call.id,
                type="function",
                function=FunctionCallParam(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in message.tool_calls or []
        ],
    )


# --- Main agent loop -----------------------------------------------

def run_agent(user_message: str) -> str:
    settings = get_settings()
    openai_client = get_openai_client()
    openai_model = settings.openai_model

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=user_message),
    ]
    tool_schemas = get_tool_schemas(settings)
    offered_tool_names = {schema["function"]["name"] for schema in tool_schemas}
    budget = ToolBudget.from_settings(settings)
    deadline = time.monotonic() + settings.agent_timeout_seconds

    for _ in range(settings.agent_max_steps):
        # The deadline is checked between rounds; a model call already in flight
        # is bounded only by the OpenAI client timeout.
        if time.monotonic() >= deadline:
            logger.warning("Agent deadline (%.0f s) reached", settings.agent_timeout_seconds)
            break

        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=messages,
            tools=tool_schemas,
        )
        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        # The model wants to call one or more tools -> run them and send the results back.
        messages.append(to_assistant_message_param(assistant_message))
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            refusal = budget.refusal(tool_name)
            if tool_name not in offered_tool_names:
                # Covers made-up names and tools that exist but are not configured.
                tool_result = f"Unknown tool: {tool_name}"
            elif refusal is not None:
                tool_result = refusal
            else:
                budget.record(tool_name)
                tool_result = execute_tool(tool_name, tool_call.function.arguments)

            # Every tool_call id must get a "tool" message, otherwise the API rejects the next call.
            messages.append(
                ChatCompletionToolMessageParam(role="tool", tool_call_id=tool_call.id, content=tool_result)
            )

    # Step or time limit reached: force a final text answer by not offering tools anymore.
    logger.warning("Agent stopped using tools (step limit %d or deadline)", settings.agent_max_steps)
    messages.append(ChatCompletionSystemMessageParam(role="system", content=STEP_LIMIT_PROMPT))
    final_response = openai_client.chat.completions.create(model=openai_model, messages=messages)
    return final_response.choices[0].message.content or ""
