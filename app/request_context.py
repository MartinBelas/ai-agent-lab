"""
Request ID (correlation ID) for tracing one request across services.

The Java AI-Demo app sends a short ID in the X-Request-ID header with every call
and logs it. This service:
1. takes the ID from the header, or generates its own when it is missing or not
   valid (e.g. when the service is called directly with curl),
2. makes it available on every log record as record.request_id -- including the
   httpx records for the OpenAI and Tavily calls -- so one request can be found in
   the logs of both services,
3. returns it in the X-Request-ID response header,
4. forwards it when it calls AI-Demo back (get_ai_demo_providers). It is never
   sent to third parties (OpenAI, Tavily).

The ID lives in a context variable, so no function has to pass it around as a
parameter. FastAPI runs sync endpoints in a worker thread with a copy of the
current context, so the value set by the middleware is visible there too.
"""

import logging
import re
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"

# Shown in log lines written outside of a request (startup, tests...).
NO_REQUEST_ID = "-"

# The header comes from the caller and ends up in log lines, so only a short,
# harmless set of characters is accepted: no spaces, newlines or control
# characters (log injection), no huge values. A UUID fits, so does the 8-character
# ID AI-Demo sends.
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_request_id: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)

logger = logging.getLogger(__name__)


def get_request_id() -> str | None:
    """The ID of the request being handled, or None outside of a request."""
    request_id = _request_id.get()
    return None if request_id == NO_REQUEST_ID else request_id


def new_request_id() -> str:
    """Same shape as the IDs AI-Demo generates: 8 hex characters."""
    return uuid.uuid4().hex[:8]


def is_valid_request_id(value: str) -> bool:
    return VALID_REQUEST_ID.fullmatch(value) is not None


def resolve_request_id(header_value: str | None) -> str:
    """The caller's ID if it is valid, otherwise a new one."""
    if header_value is not None and is_valid_request_id(header_value):
        return header_value
    return new_request_id()


def install_request_id_log_records() -> None:
    """
    Adds the current request ID to every log record as record.request_id, so any
    log format can use %(request_id)s.

    It wraps the log record factory instead of adding a filter to one handler: the
    attribute is then set on records from every logger (including httpx) and is
    available to every handler -- also when the logging was configured by someone
    else (uvicorn --log-config, Cloud Run, pytest). It is also captured when the
    record is created, i.e. in the request's own context. Safe to call repeatedly.
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "adds_request_id", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = current_factory(*args, **kwargs)
        record.request_id = _request_id.get()
        return record

    factory.adds_request_id = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class RequestIdMiddleware:
    """
    Plain ASGI middleware (not @app.middleware("http")): it sets the context
    variable for the whole request and adds the response header, also to error
    responses produced by exception handlers.

    An exception no handler takes care of would otherwise become a 500 sent by
    Starlette's ServerErrorMiddleware. That one sits outside this middleware: it
    runs after the context has been reset and sends through the original `send`,
    i.e. without the header. So this middleware sends that 500 itself.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Headers: a typed, case-insensitive view of the raw ASGI header list.
        header_value = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = resolve_request_id(header_value)
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        token = _request_id.set(request_id)
        try:
            if header_value is not None and header_value != request_id:
                # Logged only after the context variable is set, so the record carries
                # the new ID returned to the caller. The rejected value itself is not
                # logged -- that is what must not reach the log.
                logger.warning(
                    "Ignoring invalid %s header (%d characters), using %s instead",
                    REQUEST_ID_HEADER,
                    len(header_value),
                    request_id,
                )
            await self.app(scope, receive, send_with_request_id)
        except Exception as error:
            # Logged here, in the request's context, so the traceback carries the ID.
            # The server logs it again once it is re-raised, but only with "-".
            logger.exception("Unhandled %s, responding with 500", type(error).__name__)
            if not response_started:
                # The same response ServerErrorMiddleware would send. It then sees
                # that the response has started and only re-raises.
                await send_with_request_id(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    }
                )
                await send_with_request_id({"type": "http.response.body", "body": b"Internal Server Error"})
            raise
        finally:
            _request_id.reset(token)
