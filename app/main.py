"""
AI Agent Lab -- a minimal FastAPI microservice for experimenting
with OpenAI tool calling.

Run locally (configuration is read from .env, see .env.example):
    uvicorn app.main:app --reload --port 8081

Test:
    curl http://localhost:8081/api/health
    curl -X POST http://localhost:8081/api/chat -H "Content-Type: application/json" \\
         -d '{"message": "What time is it now?"}'

The /api/chat contract (v1) is meant for the Java AI-Demo app, which calls this
service server-to-server; see "API contract" in README.md. Every request carries an
X-Request-ID (sent by AI-Demo or generated here) that appears in all log lines.
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from openai import OpenAIError
from pydantic import BaseModel, Field, model_validator

from app.agent import AgentResult, HistoryMessage, run_agent
from app.request_context import RequestIdMiddleware, install_request_id_log_records

# Every line carries the request ID, the same one AI-Demo logs (see app/request_context.py).
LOG_FORMAT = "%(levelname)s:     [%(request_id)s] %(name)s - %(message)s"


def configure_logging() -> None:
    """
    Every log record gets the request ID (record.request_id), however the logging
    is configured -- a custom configuration can use %(request_id)s too.

    Uvicorn configures only its own loggers. When nobody has configured the root
    logger, add a console handler with LOG_FORMAT; without it, INFO logs from our
    modules (app.*) and from httpx (the outgoing requests to OpenAI and Tavily)
    would not show up. An existing root configuration (uvicorn --log-config, pytest)
    is left alone, apart from the request ID attribute.
    """
    install_request_id_log_records()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


configure_logging()

logger = logging.getLogger(__name__)

# Anything that turns a user message (plus earlier conversation) into a result.
# The endpoint depends on this contract, not on a concrete agent implementation.
AgentRunner = Callable[[str, Sequence[HistoryMessage]], AgentResult]

MAX_MESSAGE_LENGTH = 4000
# The history is resent with every request and paid for as input tokens, so it is
# capped both by count and by total size. The caller is expected to send only the
# most recent messages.
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_TOTAL_CHARS = 16000

app = FastAPI(title="AI Agent Lab", version="0.3.0")
app.add_middleware(RequestIdMiddleware)


class ChatHistoryMessage(BaseModel):
    # Only user/assistant: a caller must not be able to send a system prompt or tool results.
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    # Validated by Pydantic before the request reaches the agent -> 422 otherwise.
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
        description="Earlier messages of the conversation, oldest first. Optional.",
    )

    @model_validator(mode="after")
    def history_is_not_too_long(self) -> "ChatRequest":
        total_chars = sum(len(message.content) for message in self.history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            raise ValueError(f"history must not exceed {MAX_HISTORY_TOTAL_CHARS} characters in total")
        return self


class ChatSource(BaseModel):
    title: str
    url: str


class ChatUsage(BaseModel):
    model_calls: int = Field(description="Number of OpenAI calls made for this request.")
    tool_calls: int = Field(description="Number of tool executions (all tools).")
    web_searches: int = Field(description="Number of web searches (each one costs a Tavily credit).")
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    reply: str
    sources: list[ChatSource] = Field(
        description="Web pages returned by web_search during this request (http/https only, deduplicated)."
    )
    usage: ChatUsage


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
    # Sizes and counters only -- chat content is never logged.
    logger.info("Chat request (history=%d)", len(chat_request.history))
    start = time.monotonic()
    history = [HistoryMessage(role=message.role, content=message.content) for message in chat_request.history]
    result = agent(chat_request.message, history)
    usage = result.usage
    logger.info(
        "Chat reply in %d ms (model_calls=%d, tool_calls=%d, web_searches=%d, tokens=%d/%d, sources=%d)",
        round((time.monotonic() - start) * 1000),
        usage.model_calls,
        usage.tool_calls,
        usage.web_searches,
        usage.input_tokens,
        usage.output_tokens,
        len(result.sources),
    )
    return ChatResponse(
        reply=result.reply,
        sources=[ChatSource(title=source.title, url=source.url) for source in result.sources],
        usage=ChatUsage(
            model_calls=usage.model_calls,
            tool_calls=usage.tool_calls,
            web_searches=usage.web_searches,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
    )
