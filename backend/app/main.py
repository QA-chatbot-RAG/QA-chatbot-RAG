import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models import ChatRequest, ChatResponse, Latency
from app.providers import (
    Embedder,
    Generator,
    HttpEmbedder,
    HttpGenerator,
    HttpReranker,
    MockGenerator,
    MockEmbedder,
    MockReranker,
    MockVectorSearch,
    PgVectorSearch,
    ProviderError,
    RaonGenerator,
    Reranker,
    VectorSearch,
)


@dataclass
class Settings:
    embedder_mode: str = "mock"
    vector_search_mode: str = "mock"
    reranker_mode: str = "off"
    generator_mode: str = "mock"
    embedder_url: str = ""
    reranker_url: str = ""
    generator_url: str = ""
    db_url: str = ""
    provider_timeout: float = 10.0
    generator_timeout: float = 120.0
    raon_api_key: str = ""
    raon_base_url: str = "https://raon-api.krafton.ai/v1"
    raon_model: str = "raon-1-mini-preview"
    raon_max_tokens: int = 4096
    raon_temperature: float = 1.0
    rerank_candidates: int = 20
    rerank_min_score: float = 0.2
    embed_dim: int = 768
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            embedder_mode=os.getenv("EMBEDDER_MODE", "mock").lower(),
            vector_search_mode=os.getenv("VECTOR_SEARCH_MODE", "mock").lower(),
            reranker_mode=os.getenv("RERANKER_MODE", "off").lower(),
            generator_mode=os.getenv("GENERATOR_MODE", "mock").lower(),
            embedder_url=os.getenv("EMBEDDER_URL", ""),
            reranker_url=os.getenv("RERANKER_URL", ""),
            generator_url=os.getenv("GENERATOR_URL", ""),
            db_url=os.getenv("DB_URL", ""),
            provider_timeout=float(os.getenv("PROVIDER_TIMEOUT", "10")),
            generator_timeout=float(os.getenv("GENERATOR_TIMEOUT", "120")),
            raon_api_key=os.getenv("RAON_API_KEY", ""),
            raon_base_url=os.getenv(
                "RAON_BASE_URL", "https://raon-api.krafton.ai/v1"
            ),
            raon_model=os.getenv("RAON_MODEL", "raon-1-mini-preview"),
            raon_max_tokens=int(os.getenv("RAON_MAX_TOKENS", "4096")),
            raon_temperature=float(os.getenv("RAON_TEMPERATURE", "1.0")),
            rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
            rerank_min_score=float(os.getenv("RERANK_MIN_SCORE", "0.2")),
            embed_dim=int(os.getenv("EMBED_DIM", "768")),
            debug=os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"},
        )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    vector_search: VectorSearch | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    embedders = {
        "mock": lambda: MockEmbedder(config.embed_dim),
        "http": lambda: HttpEmbedder(config.embedder_url, config.provider_timeout),
    }
    vector_searches = {
        "mock": MockVectorSearch,
        "pgvector": lambda: PgVectorSearch(config.db_url, config.provider_timeout),
    }
    rerankers = {
        "mock": MockReranker,
        "http": lambda: HttpReranker(config.reranker_url, config.provider_timeout),
    }
    generators = {
        "mock": MockGenerator,
        "http": lambda: HttpGenerator(config.generator_url, config.generator_timeout),
        "raon": lambda: RaonGenerator(
            api_key=config.raon_api_key,
            base_url=config.raon_base_url,
            model=config.raon_model,
            max_tokens=config.raon_max_tokens,
            temperature=config.raon_temperature,
            timeout=config.generator_timeout,
        ),
    }
    if config.embedder_mode not in embedders:
        raise ValueError("EMBEDDER_MODE must be 'mock' or 'http'")
    if config.vector_search_mode not in vector_searches:
        raise ValueError("VECTOR_SEARCH_MODE must be 'mock' or 'pgvector'")
    if config.reranker_mode not in {"off", *rerankers}:
        raise ValueError("RERANKER_MODE must be 'off', 'mock', or 'http'")
    if config.generator_mode not in generators:
        raise ValueError("GENERATOR_MODE must be 'mock', 'http', or 'raon'")
    if config.embedder_mode == "http" and not config.embedder_url:
        raise ValueError("EMBEDDER_URL is required when EMBEDDER_MODE=http")
    if config.vector_search_mode == "pgvector" and not config.db_url:
        raise ValueError("DB_URL is required when VECTOR_SEARCH_MODE=pgvector")
    if config.reranker_mode == "http" and not config.reranker_url:
        raise ValueError("RERANKER_URL is required when RERANKER_MODE=http")
    if config.generator_mode == "http" and not config.generator_url:
        raise ValueError("GENERATOR_URL is required when GENERATOR_MODE=http")
    if config.generator_mode == "raon" and not config.raon_api_key:
        raise ValueError("RAON_API_KEY is required when GENERATOR_MODE=raon")
    if config.provider_timeout <= 0:
        raise ValueError("PROVIDER_TIMEOUT must be greater than zero")
    if config.generator_timeout <= 0:
        raise ValueError("GENERATOR_TIMEOUT must be greater than zero")
    if config.rerank_candidates < 1:
        raise ValueError("RERANK_CANDIDATES must be greater than zero")
    if not 0 <= config.rerank_min_score <= 1:
        raise ValueError("RERANK_MIN_SCORE must be between zero and one")
    if config.raon_max_tokens < 1:
        raise ValueError("RAON_MAX_TOKENS must be greater than zero")
    if config.embed_dim < 1:
        raise ValueError("EMBED_DIM must be greater than zero")

    active_embedder = embedder or embedders[config.embedder_mode]()
    active_vector_search = vector_search or vector_searches[config.vector_search_mode]()
    active_reranker = (
        reranker
        if reranker is not None
        else rerankers[config.reranker_mode]()
        if config.reranker_mode != "off"
        else None
    )
    active_generator = generator or generators[config.generator_mode]()
    app = FastAPI(title="RAG QA API", version="1.0.0")

    @app.exception_handler(ProviderError)
    async def provider_error_handler(
        request: Request, exc: ProviderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "stage": exc.stage,
                    "message": f"{exc.stage.capitalize()} service failed: {exc}",
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "stages": {
                "embedder": config.embedder_mode,
                "vector_search": config.vector_search_mode,
                "reranker": config.reranker_mode,
                "generator": config.generator_mode,
            },
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse | JSONResponse:
        started = time.perf_counter()
        embed_started = time.perf_counter()
        vectors = await active_embedder.embed([body.question])
        embed_ms = round((time.perf_counter() - embed_started) * 1000)

        search_started = time.perf_counter()
        search_top_k = (
            config.rerank_candidates if active_reranker is not None else body.top_k
        )
        sources = await active_vector_search.search(vectors[0], search_top_k)
        search_ms = round((time.perf_counter() - search_started) * 1000)

        rerank_ms = 0
        if active_reranker is not None and sources:
            rerank_started = time.perf_counter()
            sources = await active_reranker.rerank(body.question, sources)
            sources = [
                source
                for source in sources
                if source.score >= config.rerank_min_score
            ]
            sources = sources[: body.top_k]
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000)

        generation_ms = 0
        if sources:
            generation_started = time.perf_counter()
            generated = await active_generator.generate(body.question, sources)
            generation_ms = round((time.perf_counter() - generation_started) * 1000)
            answer = generated.answer
            no_answer = generated.no_answer
            model = generated.model
            thinking = generated.thinking if config.debug else None
        else:
            answer = "문서에서 답을 찾지 못했습니다."
            no_answer = True
            model = "none"
            thinking = None

        result = ChatResponse(
            answer=answer,
            no_answer=no_answer,
            sources=sources,
            latency_ms=Latency(
                embed=embed_ms,
                search=search_ms,
                rerank=rerank_ms,
                generation=generation_ms,
                total=round((time.perf_counter() - started) * 1000),
            ),
            model=model,
            thinking=thinking,
        )
        content = result.model_dump(exclude={"thinking"} if not config.debug else set())
        return JSONResponse(content=content)

    return app


app = create_app()
