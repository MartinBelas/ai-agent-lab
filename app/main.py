"""
AI Agent Lab -- a minimal FastAPI microservice for experimenting
with OpenAI tool calling (lesson 1: "AI API and the first agent").

Run locally (configuration is read from .env, see .env.example):
    uvicorn app.main:app --reload --port 8081

Test:
    curl http://localhost:8081/api/health
    curl -X POST http://localhost:8081/api/chat -H "Content-Type: application/json" \
         -d '{"message": "What time is it now?"}'
"""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.agent import run_agent

# Uvicorn configures only its own loggers. Without this, INFO logs from our modules
# (app.*) and from httpx (the outgoing requests to OpenAI and Tavily) would not show up.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")

logger = logging.getLogger(__name__)

# Anything that turns a user message into a reply. The endpoint depends on this
# contract, not on a concrete agent implementation.
AgentRunner = Callable[[str], str]

MAX_MESSAGE_LENGTH = 4000

app = FastAPI(title="AI Agent Lab", version="0.1.0")


class ChatRequest(BaseModel):
    # Validated by Pydantic before the request reaches the agent -> 422 otherwise.
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    reply: str


def get_agent() -> AgentRunner:
    """
    Dependency provider for the chat endpoint (FastAPI dependency injection).
    Tests replace it via app.dependency_overrides; later it can choose between
    several agent implementations.
    """
    return run_agent


@app.exception_handler(OpenAIError)
def handle_openai_error(request: Request, error: OpenAIError) -> JSONResponse:
    """
    The upstream AI provider failed (timeout, rate limit, outage, bad key...).
    Log the details, but return only a generic message -- never the provider's
    raw error, which may contain internal information.
    """
    logger.error("OpenAI call failed: %s: %s", type(error).__name__, error)
    return JSONResponse(status_code=502, content={"detail": "The AI provider is currently unavailable."})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The service has no web page of its own, so send browsers to the API docs."""
    return RedirectResponse(url="/docs")


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "UP"}


@app.post("/api/chat")
def handle_chat_request(
    chat_request: ChatRequest,
    agent: Annotated[AgentRunner, Depends(get_agent)],
) -> ChatResponse:
    reply = agent(chat_request.message)
    return ChatResponse(reply=reply)
