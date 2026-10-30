import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = 1024


def load_model(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def reject_empty_texts(cls, texts: list[str]) -> list[str]:
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings")
        return texts


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.embedding_model = load_model(MODEL_NAME, device)
    app.state.device = device
    yield


app = FastAPI(title="RAG Embedding API", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, str | int]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "device": request.app.state.device,
    }


@app.post("/embed")
def embed(body: EmbedRequest, request: Request) -> dict[str, Any]:
    vectors = request.app.state.embedding_model.encode(
        body.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
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
async def rerank() -> None:
    raise HTTPException(status_code=501, detail="Not Implemented")
