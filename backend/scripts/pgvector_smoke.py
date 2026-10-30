import asyncio
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from app.main import Settings
from app.providers import PgVectorSearch

# 이 단락의 임베딩을 질문 벡터 대신 써서, 자기 자신이 1등(score ≈ 1.0)으로 나오는지 확인
PROBE_DOCID = "315#0"


async def run_smoke() -> None:
    project_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(project_env)
    settings = Settings.from_env()
    if not settings.db_url:
        raise SystemExit("DB_URL is missing from .env or the environment.")

    with psycopg.connect(settings.db_url) as conn:
        row = conn.execute(
            "SELECT embedding::text FROM passage_embeddings WHERE docid = %s",
            (PROBE_DOCID,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"{PROBE_DOCID} has no embedding in passage_embeddings.")
    vector = [float(value) for value in row[0].strip("[]").split(",")]

    search = PgVectorSearch(settings.db_url, settings.provider_timeout)
    started = time.perf_counter()
    sources = await search.search(vector, settings.rerank_candidates)
    elapsed = time.perf_counter() - started

    print(f"probe: {PROBE_DOCID} (dim={len(vector)})")
    print(f"results: {len(sources)}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    for rank, source in enumerate(sources[:5], start=1):
        print(f"{rank}. {source.score:.4f} {source.doc_id} {source.title} | {source.text[:40]}")
    if not sources or sources[0].doc_id != PROBE_DOCID:
        raise SystemExit("FAIL: probe passage is not ranked first.")
    print("OK")


if __name__ == "__main__":
    asyncio.run(run_smoke())
