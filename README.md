# QA-chatbot-RAG

MIRACL 한국어 의료 코퍼스 기반 RAG QA 챗봇 (Streamlit + FastAPI + pgvector + Raon)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `data/raw/` | MIRACL 한국어 의료 원본 데이터 (parquet / csv) |
| `db/schema_medical.sql` | PostgreSQL 테이블 정의 (pgvector 사용) |
| `db/seed/` | 원본 parquet을 DB에 넣기 좋게 바꾼 CSV |
| `db/medical_rag.dump` | 데이터 + 임베딩까지 다 들어간 DB 백업 파일 |
| `notebooks/embed_bge_m3.ipynb` | Colab GPU로 단락 임베딩 만드는 노트북 (BGE-M3, 1024차원) |

## 데이터 규모

| 테이블 | 개수 | 내용 |
|---|---|---|
| `articles` | 3,505 | 위키 문서 (제목, 카테고리) |
| `passages` | 16,368 | 의료 단락 (검색 대상) |
| `queries` | 52 | 평가용 질문 (train / dev) |
| `qrels` | 745 | 질문별 정답 단락 판정 |
| `passage_embeddings` | 16,368 | `BAAI/bge-m3` 임베딩 (제목 + 본문, 정규화됨) |

## 내 PC에 DB 만들기 (Windows)

필요한 것: **PostgreSQL 18 + pgvector** (설치 프로그램 Stack Builder에서 pgvector 추가)

### 방법 1. 백업 파일로 한 번에 (추천)

`db/restore_dump.bat` 더블클릭 → postgres 비밀번호 입력

- `medical_rag` DB가 만들어지고 임베딩까지 전부 들어가요.
- 이미 `medical_rag` DB가 있으면 **지우고 다시 만들어요.**
- 비밀번호 입력할 때 화면에 아무것도 안 보이는 게 정상이에요. 치고 Enter.

### 방법 2. CSV로 처음부터

1. `db/setup_from_seed.bat` 실행 → 테이블 + 데이터 (임베딩 제외)
2. Colab에서 `notebooks/embed_bge_m3.ipynb` 실행 → 구글 드라이브에 `medical_embeddings.csv` 생성
3. 그 파일을 `db/` 폴더에 넣고 `db/load_embeddings.bat` 실행

PostgreSQL 설치 위치가 다르면 실행 전에 환경변수 `PGBIN`을 bin 폴더로 지정하세요.

### Mac / Linux

```bash
createdb -E UTF8 -T template0 medical_rag
pg_restore -d medical_rag --no-owner db/medical_rag.dump
```

## 확인

```sql
SELECT count(*) FROM passage_embeddings;  -- 16368
```

## 검색할 때 주의

질문도 **반드시 같은 모델(`BAAI/bge-m3`)로, `normalize_embeddings=True`** 로 임베딩해야 해요.

```sql
SELECT p.docid, p.title, p.content, 1 - (e.embedding <=> :query_vec) AS score
FROM passage_embeddings e JOIN passages p USING (docid)
ORDER BY e.embedding <=> :query_vec
LIMIT 5;
```

## 데이터 출처

[MIRACL](https://github.com/project-miracl/miracl) 한국어 (위키백과 기반, CC BY-SA 4.0) 중 의료 관련 문서만 추린 것
