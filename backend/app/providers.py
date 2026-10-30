import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
import psycopg

from app.models import Source


logger = logging.getLogger(__name__)


class ProviderError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VectorSearch(ABC):
    @abstractmethod
    async def search(self, vector: list[float], top_k: int) -> list[Source]:
        raise NotImplementedError


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, question: str, candidates: list[Source]) -> list[Source]:
        raise NotImplementedError


class Generator(ABC):
    @abstractmethod
    async def generate(self, question: str, sources: list[Source]) -> "GenerationResult":
        raise NotImplementedError


class GenerationResult:
    def __init__(
        self, answer: str, no_answer: bool, model: str, thinking: str | None = None
    ):
        self.answer = answer
        self.no_answer = no_answer
        self.model = model
        self.thinking = thinking


class MockEmbedder(Embedder):
    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("EMBED_DIM must be greater than zero")
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.shake_256(text.encode("utf-8")).digest(self.dimension * 2)
            vector = [
                (int.from_bytes(digest[index : index + 2], "big") / 32767.5) - 1
                for index in range(0, len(digest), 2)
            ]
            vectors.append(vector)
        return vectors


class HttpEmbedder(Embedder):
    def __init__(self, url: str, timeout: float, max_length: int = 128):
        self.url = url
        self.timeout = timeout
        self.max_length = max_length

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json={"texts": texts, "max_length": self.max_length},
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data["embeddings"]
                if not isinstance(data["model"], str):
                    raise ValueError("Embedding model must be a string")
                dimension = data["dim"]
                if not isinstance(dimension, int) or dimension < 1:
                    raise ValueError("Embedding dim must be a positive integer")
                if len(embeddings) != len(texts):
                    raise ValueError("Embedding count does not match input text count")
                if any(len(vector) != dimension for vector in embeddings):
                    raise ValueError("Embedding dimensions do not match response dim")
                return [[float(value) for value in vector] for vector in embeddings]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError("embed", str(exc)) from exc


class MockVectorSearch(VectorSearch):
    _passages = [
        Source(
            doc_id="210860#0",
            title="이순신",
            text="이순신은 조선 중기의 무신으로, 임진왜란 때 조선 수군을 지휘하였다.",
            score=0.98,
        ),
        Source(
            doc_id="210860#1",
            title="이순신",
            text="이순신은 한산도 대첩과 명량 해전 등에서 승리를 거두었다.",
            score=0.95,
        ),
        Source(
            doc_id="22630#0",
            title="임진왜란",
            text="임진왜란은 1592년부터 1598년까지 조선과 일본 사이에 벌어진 전쟁이다.",
            score=0.92,
        ),
        Source(
            doc_id="13160#0",
            title="세종대왕",
            text="세종은 조선의 제4대 국왕으로 훈민정음 창제를 주도하였다.",
            score=0.89,
        ),
        Source(
            doc_id="14519#0",
            title="한글",
            text="한글은 세종대왕이 창제하여 1443년에 반포한 문자이다.",
            score=0.86,
        ),
        Source(
            doc_id="39984#0",
            title="대한민국",
            text="대한민국은 동아시아의 한반도 남부에 위치한 국가이다.",
            score=0.83,
        ),
        Source(
            doc_id="8166#0",
            title="서울",
            text="서울특별시는 대한민국의 수도이자 최대 도시이다.",
            score=0.80,
        ),
        Source(
            doc_id="19035#0",
            title="경복궁",
            text="경복궁은 조선 왕조의 법궁으로 서울특별시 종로구에 있다.",
            score=0.77,
        ),
    ]

    async def search(self, vector: list[float], top_k: int) -> list[Source]:
        return [source.model_copy() for source in self._passages[:top_k]]


class PgVectorSearch(VectorSearch):
    # 코사인 거리(<=>) 오름차순 = 가까운 단락 순. score는 코사인 유사도(1 - 거리)
    _query = """
        SELECT p.docid, p.title, p.content,
               1 - (e.embedding <=> %(vector)s::vector) AS score
        FROM passage_embeddings e
        JOIN passages p USING (docid)
        ORDER BY e.embedding <=> %(vector)s::vector
        LIMIT %(top_k)s
    """

    def __init__(self, db_url: str, timeout: float):
        self.db_url = db_url
        self.timeout = timeout

    async def search(self, vector: list[float], top_k: int) -> list[Source]:
        # psycopg 비동기 모드는 Windows 기본 이벤트 루프에서 동작하지 않아 동기 연결을 스레드에서 실행
        try:
            rows = await asyncio.to_thread(self._fetch, vector, top_k)
        except psycopg.Error as exc:
            raise ProviderError("search", str(exc)) from exc
        return [
            Source(doc_id=docid, title=title, text=content, score=float(score))
            for docid, title, content, score in rows
        ]

    def _fetch(self, vector: list[float], top_k: int) -> list[tuple[Any, ...]]:
        vector_literal = "[" + ",".join(repr(float(value)) for value in vector) + "]"
        with psycopg.connect(
            self.db_url, connect_timeout=max(1, round(self.timeout))
        ) as conn:
            with conn.cursor() as cur:
                # HNSW 인덱스는 ef_search(기본 40)개까지만 후보를 찾으므로 top_k 이상으로 맞춤
                cur.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(max(40, top_k)),),
                )
                cur.execute(self._query, {"vector": vector_literal, "top_k": top_k})
                return cur.fetchall()


class MockReranker(Reranker):
    async def rerank(self, question: str, candidates: list[Source]) -> list[Source]:
        denominator = max(len(candidates), 1)
        return [
            source.model_copy(update={"score": 1.0 - index / denominator})
            for index, source in enumerate(candidates)
        ]


class HttpReranker(Reranker):
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    async def rerank(self, question: str, candidates: list[Source]) -> list[Source]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json={
                        "question": question,
                        "candidates": [
                            {"doc_id": source.doc_id, "text": source.text}
                            for source in candidates
                        ],
                    },
                )
                response.raise_for_status()
                data = response.json()
                sources_by_id = {source.doc_id: source for source in candidates}
                reranked = []
                for result in data["results"]:
                    source = sources_by_id[result["doc_id"]]
                    reranked.append(
                        source.model_copy(update={"score": float(result["score"])})
                    )
                return reranked
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError("rerank", str(exc)) from exc


class MockGenerator(Generator):
    async def generate(self, question: str, sources: list[Source]) -> GenerationResult:
        if not sources:
            return GenerationResult(
                answer="문서에서 답을 찾지 못했습니다.",
                no_answer=True,
                model="mock-generator",
                thinking="No relevant passages were retrieved.",
            )
        return GenerationResult(
            answer=sources[0].text,
            no_answer=False,
            model="mock-generator",
            thinking=f"Selected the passage with the highest score: {sources[0].doc_id}.",
        )


class HttpGenerator(Generator):
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    async def generate(self, question: str, sources: list[Source]) -> GenerationResult:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json={
                        "question": question,
                        "context": [source.model_dump() for source in sources],
                    },
                )
                response.raise_for_status()
                data = response.json()
                return GenerationResult(
                    answer=data["answer"],
                    no_answer=data["no_answer"],
                    model=data["model"],
                    thinking=data.get("thinking"),
                )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError("generation", str(exc)) from exc


class RaonGenerator(Generator):
    _retry_delays = (2, 4, 8)
    _system_prompt = (
        "당신은 문서 기반 질의응답 도우미입니다. 반드시 아래 제공된 문서의 내용만 근거로 답하세요.\n"
        "문서에 답이 없으면 추측하지 말고 정확히 '문서에서 답을 찾을 수 없습니다.' 라고만 답하세요.\n"
        "답변에 사용한 문서는 [1], [2] 처럼 번호로 표시하세요. 한국어로 간결하게 답하세요."
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ):
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.last_finish_reason: str | None = None
        self.last_rate_limit_headers: dict[str, str] = {}

    @staticmethod
    def _error_code(response: httpx.Response) -> str | None:
        try:
            error: Any = response.json().get("error")
        except (ValueError, TypeError, AttributeError):
            return None
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code is not None else None
        return None

    async def generate(self, question: str, sources: list[Source]) -> GenerationResult:
        context = "\n".join(
            f"[{index}] ({source.title or source.doc_id}) {source.text}"
            for index, source in enumerate(sources, start=1)
        )
        user_prompt = f"{context}\n\n질문: {question}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for retry_number in range(len(self._retry_delays) + 1):
                    response = await client.post(
                        self.url, json=payload, headers=headers
                    )
                    status_code = response.status_code
                    error_code = self._error_code(response) if status_code >= 400 else None
                    if status_code == 429 or status_code >= 500:
                        if retry_number < len(self._retry_delays):
                            retry_after = response.headers.get("retry-after")
                            try:
                                delay = (
                                    float(retry_after)
                                    if retry_after
                                    else self._retry_delays[retry_number]
                                )
                            except ValueError:
                                delay = self._retry_delays[retry_number]
                            logger.warning(
                                "Raon API returned retryable HTTP %s (code=%s); "
                                "retry %s/3 in %s seconds",
                                status_code,
                                error_code,
                                retry_number + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        code_detail = f" (code: {error_code})" if error_code else ""
                        raise ProviderError(
                            "generation",
                            f"Raon API returned HTTP {status_code}{code_detail} "
                            "after 3 retries",
                        )
                    if status_code >= 400:
                        code_detail = f" (code: {error_code})" if error_code else ""
                        raise ProviderError(
                            "generation",
                            f"Raon API returned HTTP {status_code}{code_detail}",
                        )

                    try:
                        data = response.json()
                        choice = data["choices"][0]
                        message = choice["message"]
                        finish_reason = choice.get("finish_reason")
                        content = message.get("content") or ""
                        if not isinstance(content, str):
                            raise ValueError("Raon response content must be a string")
                        answer = content.strip()
                        thinking = message.get("reasoning")
                        if thinking is not None and not isinstance(thinking, str):
                            raise ValueError("Raon response reasoning must be a string")
                        self.last_finish_reason = finish_reason
                        self.last_rate_limit_headers = {
                            key: value
                            for key, value in response.headers.items()
                            if key.lower().startswith("x-ratelimit-")
                        }
                        if finish_reason == "length" and not answer:
                            logger.warning(
                                "Raon generation reached max_tokens with empty content"
                            )
                            answer = "답변이 길이 제한으로 잘렸습니다."
                            no_answer = True
                        else:
                            no_answer = answer.startswith(
                                "문서에서 답을 찾을 수 없습니다"
                            )
                        return GenerationResult(
                            answer=answer,
                            no_answer=no_answer,
                            model=data.get("model") or self.model,
                            thinking=thinking,
                        )
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                        AttributeError,
                    ) as exc:
                        raise ProviderError("generation", str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "generation",
                f"Raon API request failed ({type(exc).__name__})",
            ) from exc

        raise ProviderError("generation", "Raon API request failed unexpectedly")
