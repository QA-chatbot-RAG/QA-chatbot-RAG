-- db/seed/*.csv → 테이블 (db 폴더에서 실행)
\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;
\copy articles (article_id, title, root, category)                 FROM 'seed/articles.csv' WITH (FORMAT csv)
\copy passages (docid, article_id, passage_no, title, content)      FROM 'seed/passages.csv' WITH (FORMAT csv)
\copy queries  (qid, split, query_text, category)                   FROM 'seed/queries.csv'  WITH (FORMAT csv)
\copy qrels    (qid, docid, rel)                                    FROM 'seed/qrels.csv'    WITH (FORMAT csv)
COMMIT;
SELECT 'articles' AS 테이블, count(*) FROM articles
UNION ALL SELECT 'passages', count(*) FROM passages
UNION ALL SELECT 'queries',  count(*) FROM queries
UNION ALL SELECT 'qrels',    count(*) FROM qrels;
