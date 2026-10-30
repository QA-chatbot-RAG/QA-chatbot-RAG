-- =====================================================================
-- 의료 RAG 챗봇 DB 스키마 (PostgreSQL 15+ / pgvector)
-- 데이터: miracl_ko_medical 폴더 (MIRACL 한국어 의료 코퍼스)
--
--  파일                          → 테이블
--  medical_passages.parquet       → articles, passages        (16,368 단락 / 3,505 문서)
--  medical_queries.parquet        → queries                   (52)
--  medical_qrels.parquet          → qrels                     (745)
--  (임베딩 후 생성)               → passage_embeddings
--  (직접 작성)                    → queries(split='custom') + reference_answers
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- 위키 문서 (passages.article 값)
CREATE TABLE articles (
    article_id   TEXT PRIMARY KEY,                       -- '210860'
    title        TEXT NOT NULL,                          -- '당뇨병'
    root         TEXT NOT NULL,                          -- 의학/질병/해부학/의약품/수동 추가
    category     TEXT NOT NULL                           -- 세부 카테고리 (감염병, 증후군 ...)
);

-- 의료 단락  ← medical_passages.parquet
CREATE TABLE passages (
    docid        TEXT PRIMARY KEY,                       -- '210860#0'
    article_id   TEXT    NOT NULL REFERENCES articles(article_id),
    passage_no   INTEGER NOT NULL,                       -- '#' 뒤 번호
    title        TEXT    NOT NULL,                       -- 표시용 (articles.title과 동일)
    content      TEXT    NOT NULL,                       -- parquet의 text 컬럼
    char_len     INTEGER GENERATED ALWAYS AS (char_length(content)) STORED,
    UNIQUE (article_id, passage_no)
);

-- 임베딩 (모델 1개. e5-base=768, BGE-M3=1024 로 차원 맞출 것)
CREATE TABLE passage_embeddings (
    docid        TEXT PRIMARY KEY REFERENCES passages(docid) ON DELETE CASCADE,
    model_name   TEXT NOT NULL,                          -- 'intfloat/multilingual-e5-base'
    embedding    vector(1024) NOT NULL
);

-- 질문  ← medical_queries.parquet (+ 직접 만든 평가 질문은 split='custom')
CREATE TABLE queries (
    qid          TEXT PRIMARY KEY,
    split        TEXT NOT NULL CHECK (split IN ('train', 'dev', 'custom')),
    query_text   TEXT NOT NULL,
    category     TEXT,                                   -- 정답 단락의 카테고리
    is_unanswerable BOOLEAN NOT NULL DEFAULT FALSE       -- 기권 평가용 질문
);

-- 관련도 판정  ← medical_qrels.parquet
CREATE TABLE qrels (
    qid          TEXT     NOT NULL REFERENCES queries(qid) ON DELETE CASCADE,
    docid        TEXT     NOT NULL,                      -- 정리 과정에서 빠진 단락이 있어 FK 없음
    rel          SMALLINT NOT NULL CHECK (rel IN (0, 1)),
    PRIMARY KEY (qid, docid)
);

-- 정답 문장 (MIRACL에 없음 → 직접 작성)
CREATE TABLE reference_answers (
    qid          TEXT PRIMARY KEY REFERENCES queries(qid) ON DELETE CASCADE,
    answer_text  TEXT NOT NULL,
    evidence_docids TEXT[] NOT NULL DEFAULT '{}',        -- 근거 단락
    source       TEXT NOT NULL DEFAULT 'llm_draft' CHECK (source IN ('llm_draft', 'human_verified'))
);

-- 인덱스 (HNSW는 임베딩 적재 후 생성)
CREATE INDEX idx_passages_article  ON passages (article_id, passage_no);
CREATE INDEX idx_articles_category ON articles (category);
CREATE INDEX idx_qrels_docid       ON qrels (docid);
CREATE INDEX idx_emb_hnsw ON passage_embeddings USING hnsw (embedding vector_cosine_ops);
