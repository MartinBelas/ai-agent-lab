# AI Agent Lab

A small Python microservice for experimenting with AI agents. It is a
standalone "lab" next to the Java `ai-demo` application and runs without it;
the optional `get_ai_demo_providers` tool reads `ai-demo`'s list of LLM providers.

The learning path goes from raw tool calling, through MCP and workflow
automation, to agent frameworks.

## What it does today

- `GET /api/health` -- liveness check, always returns `{"status": "UP"}`.
- `POST /api/chat` -- sends the message (optionally with earlier conversation
  history) to an OpenAI model that can call tools, and returns the reply, the web
  pages it searched (`sources`) and usage counters. See [API contract](#api-contract).
  Tools:
  - `get_current_time` -- a trivial sample tool;
  - `web_search` -- searches the web through the [Tavily](https://tavily.com) API
    for questions that need up-to-date information. Enabled only when
    `TAVILY_API_KEY` is set;
  - `get_ai_demo_providers` -- asks the Java `ai-demo` app which LLM providers
    and models it currently offers (`GET /api/llm/providers`). Enabled only when
    `AI_DEMO_BASE_URL` is set.
- Configuration via environment variables or a local `.env` file.
- A pytest suite that never calls the real OpenAI, Tavily or `ai-demo` API.

## Project structure

```
app/
  main.py      # FastAPI endpoints (/api/health, /api/chat)
  agent.py     # Agent loop and tool definitions
  web_search.py  # web_search tool (Tavily Search API over httpx)
  ai_demo.py     # get_ai_demo_providers tool (read-only call to the Java ai-demo app)
  tool_result.py # ToolResult / Source -- what tools return to the agent loop
  prompts.py   # System prompts
  config.py    # Settings from env variables / .env (pydantic-settings)
tests/         # pytest tests, OpenAI and Tavily are mocked
requirements.txt
requirements-dev.txt   # adds pytest
pytest.ini
Dockerfile
.env.example
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then set OPENAI_API_KEY (and optionally TAVILY_API_KEY)

uvicorn app.main:app --reload --port 8081
```

Try it in the browser via the generated Swagger UI at
<http://localhost:8081/docs>, or with curl:

```bash
curl http://localhost:8081/api/health
curl -X POST http://localhost:8081/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What time is it now?"}'
```

Every `/api/chat` request calls the OpenAI API and costs a small amount. A
question that needs current information also triggers a web search, which uses
Tavily credits:

```bash
curl -X POST http://localhost:8081/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the EUR to CZK exchange rate today?"}'
```

## API contract

`POST /api/chat` (v1). The interactive schema is at `/docs`, the raw OpenAPI
document at `/openapi.json`.

Request:

```json
{
  "message": "And how much is that in USD?",
  "history": [
    {"role": "user", "content": "What is the EUR to CZK exchange rate today?"},
    {"role": "assistant", "content": "1 EUR is about 24.30 CZK."}
  ]
}
```

- `message` -- required, 1-4000 characters.
- `history` -- optional, oldest first, default `[]`. Only `user` and `assistant`
  roles are accepted, so a caller cannot inject a system prompt or tool results.
  At most 20 messages, each 1-4000 characters, 16000 characters in total. The
  service is stateless: the caller keeps the conversation and sends the recent
  part of it with every request.

Response:

```json
{
  "reply": "1 EUR is about 24.30 CZK, which is roughly ...",
  "sources": [
    {"title": "EUR/CZK exchange rate", "url": "https://example.com/eur-czk"}
  ],
  "usage": {
    "model_calls": 2,
    "tool_calls": 1,
    "web_searches": 1,
    "input_tokens": 1830,
    "output_tokens": 95
  }
}
```

- `sources` -- the web pages returned by `web_search` during this request,
  filled by code from the search results (not written by the model), deduplicated
  by URL, `http`/`https` only. Empty when no search ran. It lists what was
  searched, which is not necessarily every page the reply relies on.
- `usage` -- what the request consumed, summed over all OpenAI calls, for the
  caller's quota accounting. Token counts are `0` if the provider does not report them.

Errors: `422` invalid request (FastAPI validation details), `502` the AI provider
failed (generic message, no internal details).

## Tests

```bash
pytest -v
```

The tests replace the OpenAI client with a mock, and the Tavily API and `ai-demo`
with an `httpx.MockTransport`, so they are free, fast and do not need real API keys.
They run in a temporary directory, so your local `.env` is never used.

## Known limitations

- The agent loop is bounded: at most `AGENT_MAX_STEPS` model calls with tools and
  `AGENT_MAX_TOOL_CALLS` tool executions per request (configurable, see
  `.env.example`); after that the model must answer with what it has.
- `web_search` has its own, stricter limit: at most `WEB_SEARCH_MAX_CALLS` (default 2)
  searches per request, each returning at most `WEB_SEARCH_MAX_RESULTS` (default 3)
  results with excerpts cut to 800 characters. When the Tavily quota is used up,
  the model is told to answer without live data instead of the request failing.
- Search results are untrusted web content. They are labelled as data for the
  model, which reduces but does not remove the prompt-injection risk, so the tools
  stay read-only.
- `get_ai_demo_providers` is the only call from this service to `ai-demo`: a
  read-only GET of a public metadata endpoint, with no user data sent. Redirects
  are not followed and the output is cut to 10 providers with short names.
- Tool failures (unknown tool, invalid arguments, exceptions) are reported back to
  the model as error messages instead of failing the request.
- OpenAI errors (timeout, outage, rate limit) return `502` with a generic message.
- `AGENT_TIMEOUT_SECONDS` is an overall time budget checked between model calls; a
  call already in flight is bounded only by `OPENAI_TIMEOUT_SECONDS` (x retries).
- No token budget yet (token usage is reported, not limited). Conversation history
  is plain text only: tool results and sources from earlier turns are not replayed.

## Deploying to Cloud Run (optional)

The `Dockerfile` is Cloud Run compatible (listens on `$PORT`, default 8080):

1. Build and push the image (e.g. `docker build` and push to Artifact Registry
   or GHCR, or a GitHub Actions workflow).
2. Provide `OPENAI_API_KEY` (and `TAVILY_API_KEY`) from Secret Manager as
   environment variables, and set `AI_DEMO_BASE_URL` to the deployed `ai-demo` URL.
3. Deploy it as a separate Cloud Run service (e.g. `ai-agent-lab`), ideally
   not publicly accessible.

## Roadmap

1. Done: **AI API and the first agent** -- tool calling without a framework.
2. Done: **A robust agent loop** -- multiple tool rounds, step and tool-call
   limits, tool error handling.
3. Done: **Web search tool** (Tavily) -- answers that need up-to-date information.
4. Done: **Chat API contract v1** -- conversation history, structured sources and
   usage counters, ready for the `ai-demo` integration.
5. **Model Context Protocol** -- expose the tools as an MCP server so that
   other clients can use them too.
6. **Workflow automation with n8n** -- call this service as an HTTP step.
7. **Agent frameworks** -- LangChain/LangGraph, OpenAI Agents SDK, compared on
   the same use case.
8. **Integration with `ai-demo`** -- a private, authenticated call from the
   Java service that keeps its rate limiting and quotas in place.
