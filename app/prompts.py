"""
Prompts used by the agent.

Kept in code (not in configuration) on purpose: a prompt is part of the agent's
behaviour, it is reviewed and versioned together with the code and the tests
that depend on it, and changing it should go through a commit, not an env var.
"""

SYSTEM_PROMPT = (
    "You are a concise assistant. Use the available tools when they help "
    "answer accurately. If a tool returns an error, do not retry it endlessly; "
    "explain what you could not find out."
)

STEP_LIMIT_PROMPT = (
    "The tool usage limit for this request has been reached. "
    "Answer now using only the information you already have."
)
