"""
Application configuration loaded from environment variables and the .env file.

pydantic-settings reads the values in this order (first one wins):
1. real environment variables (e.g. set by Cloud Run or the shell),
2. the .env file in the working directory (local development),
3. defaults defined below.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required -- no default. SecretStr hides the value in repr()/logs ("**********").
    openai_api_key: SecretStr
    openai_model: str = "gpt-5.4-mini"


@lru_cache
def get_settings() -> Settings:
    """Creates Settings once, on first use (not at import time)."""
    return Settings()
