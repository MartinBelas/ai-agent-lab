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

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import run_agent

MAX_MESSAGE_LENGTH = 4000

app = FastAPI(title="AI Agent Lab", version="0.1.0")


class ChatRequest(BaseModel):
    # Validated by Pydantic before the request reaches the agent -> 422 otherwise.
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    reply: str


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "UP"}


@app.post("/api/chat")
def handle_chat_request(chat_request: ChatRequest) -> ChatResponse:
    reply = run_agent(chat_request.message)
    return ChatResponse(reply=reply)
