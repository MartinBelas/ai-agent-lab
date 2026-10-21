"""
The get_ai_demo_providers tool: asks the Java AI-Demo app which LLM providers it
currently offers (GET /api/llm/providers).

This is the only direction in which the agent calls AI-Demo: a small, read-only
request to a public metadata endpoint. The tool never sends user data there and
never calls any other AI-Demo endpoint. Only the request ID of the current chat
request is forwarded (X-Request-ID, see app/request_context.py).

Response shape (see openapi.yaml in the ai-demo repo):
    {"providers": [{"id": "OPENAI", "model": "gpt-5.4-mini"}, ...]}
"""

import logging
from functools import lru_cache
from typing import Any

import httpx

from app.config import get_settings
from app.request_context import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger(__name__)

PROVIDERS_PATH = "/api/llm/providers"

# The response comes from our own service, but it is still external input to the
# prompt, so it is kept small: a handful of providers with short names.
MAX_PROVIDERS = 10
MAX_FIELD_CHARS = 80


@lru_cache
def get_http_client() -> httpx.Client:
    """One shared HTTP client, created on first use. Redirects are not followed."""
    return httpx.Client(timeout=get_settings().ai_demo_timeout_seconds)


def get_ai_demo_providers() -> str:
    """Returns the LLM providers AI-Demo offers, formatted for the model."""
    base_url = get_settings().ai_demo_base_url
    if base_url is None:
        # The agent does not offer the tool without a URL, so this is a safety net.
        return "Error: the AI-Demo connection is not configured."

    try:
        response = get_http_client().get(base_url + PROVIDERS_PATH, headers=request_id_headers())
    except httpx.TimeoutException:
        logger.warning("AI-Demo providers request timed out")
        return "Error: the AI-Demo app did not respond in time."
    except httpx.TransportError as error:
        # Connection refused, DNS failure... -- AI-Demo is down or the URL is wrong.
        logger.warning("AI-Demo providers request failed: %s", type(error).__name__)
        return "Error: the AI-Demo app is not reachable right now."

    # Any non-2xx (including a redirect) is unexpected: raise, and execute_tool logs
    # it and reports only the error type to the model.
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        logger.warning("AI-Demo providers response is not JSON")
        return "Error: the AI-Demo app returned an unexpected response."
    return format_providers(payload)


def request_id_headers() -> dict[str, str]:
    """
    Forwards the ID of the current chat request, so AI-Demo can tie this call to the
    request that caused it. It is our own service; third parties never get the ID.
    """
    request_id = get_request_id()
    return {REQUEST_ID_HEADER: request_id} if request_id else {}


def format_providers(payload: Any) -> str:
    """Turns the /api/llm/providers JSON into a short list the model can quote."""
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, list):
        logger.warning("AI-Demo providers response has an unexpected shape")
        return "Error: the AI-Demo app returned an unexpected response."

    lines = []
    for provider in providers[:MAX_PROVIDERS]:
        if not isinstance(provider, dict):
            continue
        provider_id = clean(provider.get("id"))
        model = clean(provider.get("model"))
        if provider_id:
            lines.append(f"- {provider_id}" + (f" (model: {model})" if model else ""))

    logger.info("AI-Demo reports %d providers", len(lines))
    if not lines:
        return "The AI-Demo app currently has no LLM providers available."
    return "LLM providers currently available in the AI-Demo app:\n" + "\n".join(lines)


def clean(value: object) -> str:
    """A single-line, length-limited string; anything that is not a string is dropped."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_FIELD_CHARS]
