# AI Agent Lab

A small Python microservice for experimenting with AI agents. It is a
standalone "lab" next to the Java/Spring `ai-demo`
application and does not depend on it.

The learning path goes from raw tool calling, through MCP and workflow
automation, to agent frameworks.

## What it does today

- `GET /api/health` -- liveness check, always returns `{"status": "UP"}`.
- `POST /api/chat` -- sends the message to an OpenAI model that can call one
  sample tool (`get_current_time`). The message is validated (1-4000 characters).
- Configuration via environment variables or a local `.env` file.
- A pytest suite that never calls the real OpenAI API.

## Project structure

```
app/
  main.py      # FastAPI endpoints (/api/health, /api/chat)
  agent.py     # Agent loop and tool definitions
  config.py    # Settings from env variables / .env (pydantic-settings)
tests/         # pytest tests, the OpenAI client is mocked
requirements.txt
requirements-dev.txt   # adds pytest and httpx
pytest.ini
Dockerfile
.env.example
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then set OPENAI_API_KEY (loaded automatically)

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

Every `/api/chat` request calls the OpenAI API and costs a small amount.

## Tests

```bash
pytest -v
```

The tests replace the OpenAI client with a mock, so they are free, fast and
do not need a real `OPENAI_API_KEY`.

## Known limitations

- The agent performs at most one round of tool calls; it cannot chain tools yet.
- An exception inside a tool fails the whole request (covered by an `xfail` test).
- No limits yet on the number of agent steps, timeouts or token usage.

## Deploying to Cloud Run (optional)

The `Dockerfile` is Cloud Run compatible (listens on `$PORT`, default 8080):

1. Build and push the image (e.g. `docker build` and push to Artifact Registry
   or GHCR, or a GitHub Actions workflow).
2. Provide `OPENAI_API_KEY` from Secret Manager as an environment variable.
3. Deploy it as a separate Cloud Run service (e.g. `ai-agent-lab`), ideally
   not publicly accessible.

## Roadmap

1. Done: **AI API and the first agent** -- tool calling without a framework.
2. **A robust agent loop** -- multiple tool rounds, step and time limits,
   tool error handling.
3. **Model Context Protocol** -- expose the tools as an MCP server so that
   other clients can use them too.
4. **Workflow automation with n8n** -- call this service as an HTTP step.
5. **Agent frameworks** -- LangChain/LangGraph, OpenAI Agents SDK, compared on
   the same use case.
6. **Integration with `ai-demo`** -- a private, authenticated call from the
   Java service that keeps its rate limiting and quotas in place.
