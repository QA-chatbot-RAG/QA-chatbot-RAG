-- notebooks/embed_bge_m3.ipynb 결과(medical_embeddings.csv) → passage_embeddings (db 폴더에서 실행)
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
SET maintenance_work_mem = '512MB';
BEGIN;
TRUNCATE passage_embeddings;
\copy passage_embeddings (docid, model_name, embedding) FROM 'medical_embeddings.csv' WITH (FORMAT csv)
COMMIT;
REINDEX INDEX idx_emb_hnsw;
SELECT count(*) AS 임베딩_개수, (SELECT count(*) FROM passages) AS 단락_개수 FROM passage_embeddings;
