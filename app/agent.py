"""
The simplest possible "agent": an OpenAI model with one custom tool
(tool calling / function calling). The model decides on its own whether
to call the tool, based on its description in the schema.

Once this works, it's a natural foundation for:
- more tools (e.g. querying Firestore, calling an internal API)
- an MCP server (exposing the tools to another client, not just this agent)
"""

import json
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    """
    Creates the OpenAI client lazily, on the first request -- not at import time.
    Thanks to that, the module can be imported without an API key (e.g. in tests),
    and tests can replace this function with a fake client.
    """
    return OpenAI(api_key=get_settings().openai_api_key.get_secret_value())


# --- Definition of the tool the model can call -----------------------

def get_current_time() -> str:
    """Returns the current server time (UTC), just as a sample tool."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# What the model sees. "name" is the identifier the model sends back when it
# wants the tool called -- it must match the key in TOOL_HANDLERS.
TOOL_SCHEMAS = [
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
    }
]

# Translates the tool name returned by the model to the Python function.
# Also, an allowlist: only functions listed here can ever be executed.
TOOL_HANDLERS = {
    "get_current_time": get_current_time,
}


# --- Main agent loop -----------------------------------------------

def run_agent(user_message: str) -> str:
    openai_client = get_openai_client()
    openai_model = get_settings().openai_model

    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. If you need the current time, use the get_current_time tool.",
        },
        {"role": "user", "content": user_message},
    ]

    initial_response = openai_client.chat.completions.create(
        model=openai_model,
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    first_choice = initial_response.choices[0]
    tool_calls = first_choice.message.tool_calls

    if not tool_calls:
        return first_choice.message.content or ""

    # The model wants to call one or more tools -> call them and send the result back
    messages.append(first_choice.message)

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments or "{}")
        tool_function = TOOL_HANDLERS.get(tool_name)

        tool_result = tool_function(**tool_args) if tool_function else f"Unknown tool: {tool_name}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result),
            }
        )

    final_response = openai_client.chat.completions.create(model=openai_model, messages=messages)
    return final_response.choices[0].message.content or ""
