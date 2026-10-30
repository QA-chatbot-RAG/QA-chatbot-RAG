import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EMBEDDING_DIM = 1024
RERANK_MAX_LENGTH = 512


def load_model(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def load_reranker(model_name: str, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        device=device,
        max_length=RERANK_MAX_LENGTH,
        activation_fn=torch.nn.Identity(),
    )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    max_length: int | None = Field(default=None, ge=1, le=8192)

    @field_validator("texts")
    @classmethod
    def reject_empty_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings")
        return texts


class RerankCandidate(BaseModel):
    doc_id: str
    text: str


class RerankRequest(BaseModel):
    question: str = Field(min_length=1)
    candidates: list[RerankCandidate] = Field(min_length=1, max_length=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.embedding_model = load_model(MODEL_NAME, device)
    app.state.embedding_lock = threading.Lock()
    app.state.reranker = (
        load_reranker(RERANK_MODEL_NAME, device) if RERANK_ENABLED else None
    )
    app.state.device = device
    yield


app = FastAPI(title="RAG Embedding API", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "device": request.app.state.device,
        "reranker": {
            "model": RERANK_MODEL_NAME,
            "enabled": RERANK_ENABLED,
        },
    }


@app.post("/embed")
def embed(body: EmbedRequest, request: Request) -> dict[str, Any]:
    model = request.app.state.embedding_model
    with request.app.state.embedding_lock:
        original_max_length = model.max_seq_length
        try:
            if body.max_length is not None:
                model.max_seq_length = min(original_max_length, body.max_length)
            vectors = model.encode(
                body.texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        finally:
            model.max_seq_length = original_max_length
    embeddings = vectors.tolist()
    if len(embeddings) != len(body.texts) or any(
        len(vector) != EMBEDDING_DIM for vector in embeddings
    ):
        raise RuntimeError(
            f"Embedding model must return {EMBEDDING_DIM}-dimensional vectors"
        )
    return {
        "embeddings": embeddings,
        "model": MODEL_NAME,
        "dim": EMBEDDING_DIM,
    }


@app.post("/rerank")
def rerank(body: RerankRequest, request: Request) -> dict[str, list[dict[str, Any]]]:
    model = request.app.state.reranker
    if model is None:
        raise HTTPException(status_code=501, detail="Reranking is disabled")

    pairs = [(body.question, candidate.text) for candidate in body.candidates]
    logits = model.predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
        apply_softmax=False,
    )
    scores = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64).reshape(-1))
    if scores.numel() != len(body.candidates):
        raise RuntimeError("Reranker score count does not match candidate count")

    results = [
        {"doc_id": candidate.doc_id, "score": float(score)}
        for candidate, score in zip(body.candidates, scores.tolist())
    ]
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"results": results}
