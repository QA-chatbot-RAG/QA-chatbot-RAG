import math

from fastapi.testclient import TestClient

import main


class FakeEmbeddingModel:
    pass


class FakeReranker:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        apply_softmax: bool,
    ) -> list[int]:
        assert show_progress_bar is False
        assert convert_to_numpy is True
        assert apply_softmax is False
        self.pairs = pairs
        return [len(text) for _, text in pairs]


def test_rerank_sorts_results_by_sigmoid_score(monkeypatch) -> None:
    model = FakeReranker()
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(
        main, "load_model", lambda model_name, device: FakeEmbeddingModel()
    )
    monkeypatch.setattr(main, "load_reranker", lambda model_name, device: model)

    with TestClient(main.app) as client:
        response = client.post(
            "/rerank",
            json={
                "question": "질문",
                "candidates": [
                    {"doc_id": "short", "text": "짧음"},
                    {"doc_id": "long", "text": "더 긴 문장"},
                    {"doc_id": "medium", "text": "중간 문장"},
                ],
            },
        )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["doc_id"] for result in results] == [
        "long",
        "medium",
        "short",
    ]
    assert all(0.0 <= result["score"] <= 1.0 for result in results)
    assert math.isclose(results[0]["score"], 1 / (1 + math.exp(-len("더 긴 문장"))))
    assert model.pairs == [
        ("질문", "짧음"),
        ("질문", "더 긴 문장"),
        ("질문", "중간 문장"),
    ]


def test_empty_candidates_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(main, "RERANK_ENABLED", True)
    monkeypatch.setattr(
        main, "load_model", lambda model_name, device: FakeEmbeddingModel()
    )
    monkeypatch.setattr(main, "load_reranker", lambda model_name, device: object())

    with TestClient(main.app) as client:
        response = client.post(
            "/rerank", json={"question": "질문", "candidates": []}
        )

    assert response.status_code == 422


def test_rerank_disabled_does_not_load_model_and_returns_501(monkeypatch) -> None:
    monkeypatch.setattr(main, "RERANK_ENABLED", False)
    monkeypatch.setattr(
        main, "load_model", lambda model_name, device: FakeEmbeddingModel()
    )

    def fail_if_loaded(model_name: str, device: str) -> None:
        raise AssertionError("reranker must not load when disabled")

    monkeypatch.setattr(main, "load_reranker", fail_if_loaded)

    with TestClient(main.app) as client:
        assert client.get("/health").json()["reranker"] == {
            "model": "BAAI/bge-reranker-v2-m3",
            "enabled": False,
        }
        response = client.post(
            "/rerank",
            json={
                "question": "질문",
                "candidates": [{"doc_id": "1", "text": "문장"}],
            },
        )

    assert response.status_code == 501


def test_reranker_model_uses_512_token_limit(monkeypatch) -> None:
    calls = {}

    def fake_cross_encoder(model_name, *, device, max_length, activation_fn):
        calls.update(
            model_name=model_name,
            device=device,
            max_length=max_length,
            activation_fn=activation_fn,
        )
        return object()

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_cross_encoder)

    main.load_reranker("test-reranker", "cpu")

    assert calls == {
        "model_name": "test-reranker",
        "device": "cpu",
        "max_length": 512,
        "activation_fn": calls["activation_fn"],
    }
    assert isinstance(calls["activation_fn"], main.torch.nn.Identity)
