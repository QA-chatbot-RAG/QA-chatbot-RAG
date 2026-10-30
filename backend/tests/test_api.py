import asyncio

import httpx
from fastapi.testclient import TestClient

from app.main import Settings, create_app


def _mock_raon_responses(monkeypatch, responses):
    calls = []

    async def fake_post(self, url, json, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        status_code, body, response_headers = responses.pop(0)
        return httpx.Response(
            status_code,
            json=body,
            headers=response_headers,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def _raon_client(debug=False):
    return TestClient(
        create_app(
            Settings(
                generator_mode="raon",
                raon_api_key="test-key-not-a-real-credential",
                debug=debug,
            )
        )
    )


def test_health():
    client = TestClient(create_app(Settings()))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "stages": {
            "embedder": "mock",
            "vector_search": "mock",
            "reranker": "off",
            "generator": "mock",
        },
    }


def test_chat_mock_runs_all_stages_without_reranking():
    client = TestClient(create_app(Settings()))
    response = client.post("/chat", json={"question": "이순신은 누구인가요?", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == body["sources"][0]["text"]
    assert not body["answer"].startswith("이순신:")
    assert body["no_answer"] is False
    assert body["sources"][0]["doc_id"] == "210860#0"
    assert body["sources"][0]["title"] == "이순신"
    assert set(body["latency_ms"]) == {
        "embed",
        "search",
        "rerank",
        "generation",
        "total",
    }
    assert body["latency_ms"]["rerank"] == 0
    assert body["model"] == "mock-generator"
    assert "thinking" not in body


def test_chat_with_mock_reranker_searches_candidates_then_limits_results():
    client = TestClient(
        create_app(Settings(reranker_mode="mock", rerank_candidates=7))
    )
    response = client.post("/chat", json={"question": "이순신은 누구인가요?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 3
    assert body["sources"][0]["doc_id"] == "210860#0"
    assert body["sources"][0]["score"] == 1.0
    assert body["latency_ms"]["rerank"] >= 0


def test_chat_without_search_results_skips_generation():
    class EmptyVectorSearch:
        async def search(self, vector, top_k):
            return []

    client = TestClient(create_app(Settings(), vector_search=EmptyVectorSearch()))
    response = client.post("/chat", json={"question": "질문"})

    assert response.status_code == 200
    body = response.json()
    assert body["no_answer"] is True
    assert body["sources"] == []
    assert body["model"] == "none"
    assert body["latency_ms"]["generation"] == 0


def test_thinking_is_only_included_in_debug_mode():
    client = TestClient(create_app(Settings(debug=True)))
    response = client.post("/chat", json={"question": "이순신은 누구인가요?"})

    assert response.status_code == 200
    assert response.json()["thinking"]


def test_source_without_title_is_returned_with_null_title():
    from app.models import Source
    from app.providers import MockVectorSearch

    class UntitledVectorSearch(MockVectorSearch):
        async def search(self, vector, top_k):
            return [Source(doc_id="doc-1", text="본문", score=0.5)][:top_k]

    client = TestClient(
        create_app(Settings(), vector_search=UntitledVectorSearch())
    )
    response = client.post("/chat", json={"question": "질문"})

    assert response.status_code == 200
    assert response.json()["sources"][0]["title"] is None


def test_embed_provider_failure_identifies_failed_stage():
    from app.providers import ProviderError

    class BrokenEmbedder:
        async def embed(self, texts):
            raise ProviderError("embed", "upstream unavailable")

    client = TestClient(create_app(Settings(), embedder=BrokenEmbedder()))
    response = client.post("/chat", json={"question": "test"})

    assert response.status_code == 503
    assert response.json()["error"]["stage"] == "embed"


def test_raon_generator_separates_answer_and_hides_thinking_by_default(monkeypatch):
    calls = _mock_raon_responses(
        monkeypatch,
        [
            (
                200,
                {
                    "model": "raon-1-mini-preview",
                    "choices": [
                        {
                            "message": {
                                "content": " 서울입니다. ",
                                "reasoning": "문서에서 서울을 확인했습니다.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
                {},
            )
        ],
    )
    client = _raon_client()

    response = client.post("/chat", json={"question": "대한민국의 수도는 어디인가요?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "서울입니다."
    assert "thinking" not in body
    assert body["model"] == "raon-1-mini-preview"
    assert calls[0]["headers"]["Authorization"].startswith("Bearer ")
    assert calls[0]["json"]["messages"][0]["role"] == "system"
    assert "[1] (이순신)" in calls[0]["json"]["messages"][1]["content"]


def test_raon_thinking_is_included_only_in_debug_mode(monkeypatch):
    _mock_raon_responses(
        monkeypatch,
        [
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "답변",
                                "reasoning": "내부 추론",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
                {},
            )
        ],
    )

    response = _raon_client(debug=True).post(
        "/chat", json={"question": "질문"}
    )

    assert response.status_code == 200
    assert response.json()["thinking"] == "내부 추론"


def test_raon_no_answer_phrase_sets_no_answer(monkeypatch):
    _mock_raon_responses(
        monkeypatch,
        [
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "문서에서 답을 찾을 수 없습니다.",
                                "reasoning": "근거 없음",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
                {},
            )
        ],
    )

    response = _raon_client().post("/chat", json={"question": "질문"})

    assert response.status_code == 200
    assert response.json()["no_answer"] is True


def test_raon_empty_content_after_length_limit_is_handled(monkeypatch, caplog):
    _mock_raon_responses(
        monkeypatch,
        [
            (
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "", "reasoning": "긴 추론"},
                            "finish_reason": "length",
                        }
                    ]
                },
                {},
            )
        ],
    )

    response = _raon_client().post("/chat", json={"question": "질문"})

    assert response.status_code == 200
    assert response.json()["answer"] == "답변이 길이 제한으로 잘렸습니다."
    assert response.json()["no_answer"] is True
    assert "reached max_tokens" in caplog.text


def test_raon_retries_rate_limits_then_succeeds(monkeypatch):
    from app.models import Source
    from app.providers import RaonGenerator

    calls = _mock_raon_responses(
        monkeypatch,
        [
            (429, {"error": {"code": "rate_limit"}}, {"retry-after": "0"}),
            (429, {"error": {"code": "rate_limit"}}, {}),
            (
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "성공", "reasoning": ""},
                            "finish_reason": "stop",
                        }
                    ]
                },
                {},
            ),
        ],
    )
    delays = []

    async def no_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("app.providers.asyncio.sleep", no_sleep)
    generator = RaonGenerator(
        api_key="test-key",
        base_url="https://raon-api.krafton.ai/v1",
        model="raon-1-mini-preview",
        max_tokens=4096,
        temperature=1.0,
        timeout=120,
    )

    result = asyncio.run(
        generator.generate("질문", [Source(doc_id="doc", text="본문", score=1.0)])
    )

    assert result.answer == "성공"
    assert len(calls) == 3
    assert delays == [0.0, 4]


def test_raon_unauthorized_returns_generation_503_without_key(monkeypatch):
    calls = _mock_raon_responses(
        monkeypatch,
        [(401, {"error": {"code": "invalid_api_key"}}, {})],
    )

    response = _raon_client().post("/chat", json={"question": "질문"})

    assert response.status_code == 503
    assert response.json()["error"]["stage"] == "generation"
    assert "invalid_api_key" in response.json()["error"]["message"]
    assert "test-key-not-a-real-credential" not in response.text
    assert len(calls) == 1


def test_raon_token_limit_error_is_not_retried(monkeypatch):
    calls = _mock_raon_responses(
        monkeypatch,
        [(400, {"error": {"code": "token_limit_exceeded"}}, {})],
    )

    response = _raon_client().post("/chat", json={"question": "질문"})

    assert response.status_code == 503
    assert response.json()["error"]["stage"] == "generation"
    assert "token_limit_exceeded" in response.json()["error"]["message"]
    assert len(calls) == 1


def test_health_reports_raon_generator_mode():
    response = TestClient(
        create_app(
            Settings(generator_mode="raon", raon_api_key="test-key")
        )
    ).get("/health")

    assert response.status_code == 200
    assert response.json()["stages"]["generator"] == "raon"


def test_raon_mode_requires_api_key():
    import pytest

    with pytest.raises(ValueError, match="RAON_API_KEY is required"):
        create_app(Settings(generator_mode="raon"))


def _mock_pgvector(monkeypatch, rows=None, error=None):
    import psycopg

    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params):
            calls.append({"query": query, "params": params})

        def fetchall(self):
            return rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return FakeCursor()

    def fake_connect(conninfo, **kwargs):
        calls.append({"conninfo": conninfo, **kwargs})
        if error:
            raise error
        return FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    return calls


_PG_ROWS = [
    ("210860#0", "당뇨병", "당뇨병은 혈당이 높아지는 대사 질환이다.", 0.81),
    ("210860#3", "당뇨병", "대표 증상은 다뇨, 다음, 다식이다.", 0.78),
    ("1063536#0", "알부민뇨", "알부민뇨는 소변에 알부민이 섞여 나오는 것이다.", 0.52),
]


def test_pgvector_mode_requires_db_url():
    import pytest

    with pytest.raises(ValueError, match="DB_URL is required"):
        create_app(Settings(vector_search_mode="pgvector"))


def test_pgvector_search_maps_rows_to_sources(monkeypatch):
    from app.providers import PgVectorSearch

    calls = _mock_pgvector(monkeypatch, rows=_PG_ROWS)
    search = PgVectorSearch("postgresql://rag:pw@db:5432/medical_rag", timeout=10)

    sources = asyncio.run(search.search([0.5, -0.25, 1.0], top_k=3))

    assert [source.doc_id for source in sources] == ["210860#0", "210860#3", "1063536#0"]
    assert sources[1].title == "당뇨병"
    assert sources[1].text == "대표 증상은 다뇨, 다음, 다식이다."
    assert sources[1].score == 0.78
    assert calls[0]["conninfo"] == "postgresql://rag:pw@db:5432/medical_rag"
    assert calls[0]["connect_timeout"] == 10
    assert calls[2]["params"] == {"vector": "[0.5,-0.25,1.0]", "top_k": 3}


def test_pgvector_search_raises_ef_search_to_cover_top_k(monkeypatch):
    from app.providers import PgVectorSearch

    calls = _mock_pgvector(monkeypatch, rows=[])
    search = PgVectorSearch("postgresql://db", timeout=10)

    asyncio.run(search.search([0.1], top_k=10))
    asyncio.run(search.search([0.1], top_k=50))

    ef_search_values = [call["params"] for call in calls if "hnsw.ef_search" in call.get("query", "")]
    assert ef_search_values == [("40",), ("50",)]


def test_pgvector_failure_identifies_search_stage(monkeypatch):
    import psycopg

    _mock_pgvector(monkeypatch, error=psycopg.OperationalError("connection refused"))
    client = TestClient(
        create_app(Settings(vector_search_mode="pgvector", db_url="postgresql://db"))
    )
    response = client.post("/chat", json={"question": "당뇨병의 증상은?"})

    assert response.status_code == 503
    assert response.json()["error"]["stage"] == "search"


def test_chat_searches_pgvector_candidates_then_reranks_to_top_k(monkeypatch):
    calls = _mock_pgvector(monkeypatch, rows=_PG_ROWS)
    rerank_calls = []

    async def fake_post(self, url, json):
        rerank_calls.append({"url": url, "json": json})
        return httpx.Response(
            200,
            json={
                "results": [
                    {"doc_id": "210860#3", "score": 0.97},
                    {"doc_id": "210860#0", "score": 0.64},
                    {"doc_id": "1063536#0", "score": 0.02},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    client = TestClient(
        create_app(
            Settings(
                vector_search_mode="pgvector",
                db_url="postgresql://db",
                reranker_mode="http",
                reranker_url="http://model-api:8001/rerank",
                rerank_candidates=30,
            )
        )
    )
    response = client.post("/chat", json={"question": "당뇨병의 증상은?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert calls[2]["params"]["top_k"] == 30
    assert rerank_calls[0]["json"]["candidates"][1] == {
        "doc_id": "210860#3",
        "text": "대표 증상은 다뇨, 다음, 다식이다.",
    }
    assert [source["doc_id"] for source in body["sources"]] == ["210860#3", "210860#0"]
    assert body["sources"][0] == {
        "doc_id": "210860#3",
        "title": "당뇨병",
        "text": "대표 증상은 다뇨, 다음, 다식이다.",
        "score": 0.97,
    }
