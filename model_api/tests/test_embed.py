import math

import numpy as np
from fastapi.testclient import TestClient

import main


class FakeModel:
    def __init__(self) -> None:
        self.normalize_embeddings = False
        self.max_seq_length = 512
        self.encoded_max_seq_length = None

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.normalize_embeddings = normalize_embeddings
        self.encoded_max_seq_length = self.max_seq_length
        assert convert_to_numpy is True
        return np.full((len(texts), main.EMBEDDING_DIM), 1 / 32, dtype=np.float32)


def test_health_and_embedding_contract(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(main, "load_model", lambda model_name, device: model)
    monkeypatch.setattr(
        main, "load_reranker", lambda model_name, device: object()
    )

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model": "BAAI/bge-m3",
            "dim": 1024,
            "device": "cuda" if main.torch.cuda.is_available() else "cpu",
            "reranker": {
                "model": "BAAI/bge-reranker-v2-m3",
                "enabled": True,
            },
        }

        response = client.post("/embed", json={"texts": ["질문1", "질문2"]})

    assert response.status_code == 200
    result = response.json()
    assert result["model"] == "BAAI/bge-m3"
    assert result["dim"] == 1024
    assert len(result["embeddings"]) == 2
    assert all(len(vector) == 1024 for vector in result["embeddings"])
    assert all(
        math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)
        for vector in result["embeddings"]
    )
    assert model.normalize_embeddings is True


def test_empty_text_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(main, "load_model", lambda model_name, device: FakeModel())
    monkeypatch.setattr(
        main, "load_reranker", lambda model_name, device: object()
    )

    with TestClient(main.app) as client:
        response = client.post("/embed", json={"texts": [""]})

    assert response.status_code == 422


def test_empty_batch_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(main, "load_model", lambda model_name, device: FakeModel())
    monkeypatch.setattr(
        main, "load_reranker", lambda model_name, device: object()
    )

    with TestClient(main.app) as client:
        response = client.post("/embed", json={"texts": []})

    assert response.status_code == 422


def test_embedding_request_limits_and_restores_max_sequence_length(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(main, "load_model", lambda model_name, device: model)
    monkeypatch.setattr(main, "load_reranker", lambda model_name, device: object())

    with TestClient(main.app) as client:
        response = client.post(
            "/embed",
            json={"texts": ["질문"], "max_length": 128},
        )

    assert response.status_code == 200
    assert model.encoded_max_seq_length == 128
    assert model.max_seq_length == 512
