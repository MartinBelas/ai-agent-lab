"""
Application configuration loaded from environment variables and the .env file.

pydantic-settings reads the values in this order (first one wins):
1. real environment variables (e.g. set by Cloud Run or the shell),
2. the .env file in the working directory (local development),
3. defaults defined below.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required -- no default. SecretStr hides the value in repr()/logs ("**********").
    openai_api_key: SecretStr
    openai_model: str = "gpt-5.4-mini"
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)

    # Agent loop limits (see app/agent.py).
    # How many times the model may be called *with tools available* in one request.
    agent_max_steps: int = Field(default=5, ge=1)
    # How many tool executions are allowed in one request, across all steps.
    agent_max_tool_calls: int = Field(default=8, ge=1)
    # Overall time budget for one request. Checked between model calls, so the real
    # worst case is this value plus one OpenAI call (timeout x retries).
    agent_timeout_seconds: float = Field(default=60.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Creates Settings once, on first use (not at import time)."""
    # Required fields are filled from env vars / .env at runtime, which static
    # type checkers cannot see -- hence the ignore.
    return Settings()  # type: ignore[call-arg]
