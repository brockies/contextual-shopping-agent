CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS product_embeddings (
  product_id TEXT PRIMARY KEY,
  category   TEXT,
  embedding  vector(1536),
  eco_score  INT,
  price_gbp  NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_product_embeddings_category ON product_embeddings(category);
