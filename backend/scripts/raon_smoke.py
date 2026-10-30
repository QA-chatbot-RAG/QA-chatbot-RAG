import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from app.main import Settings
from app.models import Source
from app.providers import RaonGenerator


async def run_smoke() -> None:
    project_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(project_env)
    settings = Settings.from_env()
    if not settings.raon_api_key:
        raise SystemExit("RAON_API_KEY is missing from .env or the environment.")

    generator = RaonGenerator(
        api_key=settings.raon_api_key,
        base_url=settings.raon_base_url,
        model=settings.raon_model,
        max_tokens=settings.raon_max_tokens,
        temperature=settings.raon_temperature,
        timeout=settings.generator_timeout,
    )
    sources = [
        Source(
            doc_id="raon-smoke#0",
            title="대한민국",
            text="대한민국의 수도는 서울입니다.",
            score=1.0,
        )
    ]

    started = time.perf_counter()
    result = await generator.generate("대한민국의 수도는 어디인가요?", sources)
    elapsed = time.perf_counter() - started

    print(f"answer: {result.answer}")
    print(f"thinking_length: {len(result.thinking or '')}")
    print(f"finish_reason: {generator.last_finish_reason}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    for name, value in sorted(generator.last_rate_limit_headers.items()):
        print(f"{name}: {value}")


if __name__ == "__main__":
    asyncio.run(run_smoke())
