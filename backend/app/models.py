from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class Source(BaseModel):
    doc_id: str
    text: str
    score: float
    title: str | None = None


class Latency(BaseModel):
    embed: int
    search: int
    rerank: int
    generation: int
    total: int


class ChatResponse(BaseModel):
    answer: str
    no_answer: bool
    sources: list[Source]
    latency_ms: Latency
    model: str
    thinking: str | None = None
