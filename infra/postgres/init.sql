-- ═══════════════════════════════════════════════════════════════
--  RAG Enterprise — Schema inicial PostgreSQL
--  Metadados, auditoria e versionamento
-- ═══════════════════════════════════════════════════════════════

-- Extensões
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────────────────────
--  CAMADA: Controle de datasets ingeridos
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_datasets (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(255) NOT NULL,
    domain      VARCHAR(100) NOT NULL,
    source_url  TEXT,
    version     VARCHAR(50)  NOT NULL DEFAULT '1.0.0',
    description TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
--  CAMADA MEDALLION: Rastreamento de arquivos por camada
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_files (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id    UUID REFERENCES rag_datasets(id) ON DELETE CASCADE,
    layer         VARCHAR(10) NOT NULL CHECK (layer IN ('bronze', 'silver', 'gold')),
    bucket        VARCHAR(100) NOT NULL,
    object_key    TEXT NOT NULL,
    file_format   VARCHAR(50),
    size_bytes    BIGINT,
    row_count     INTEGER,
    checksum      VARCHAR(64),
    status        VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'done', 'error')),
    error_msg     TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at  TIMESTAMPTZ,
    UNIQUE (bucket, object_key)
);

-- ─────────────────────────────────────────────────────────────
--  DOCUMENTOS: Chunks indexados no Milvus
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_file_id    UUID REFERENCES data_files(id),
    milvus_id       BIGINT UNIQUE,
    content         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    token_count     INTEGER,
    metadata        JSONB DEFAULT '{}',
    embed_model     VARCHAR(100),
    embed_version   VARCHAR(50),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
--  EXPERIMENTOS: Registro de runs RAG
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlflow_run_id   VARCHAR(100),
    query           TEXT NOT NULL,
    retrieved_docs  JSONB DEFAULT '[]',
    prompt_used     TEXT,
    response        TEXT,
    llm_model       VARCHAR(100),
    embed_model     VARCHAR(100),
    top_k           INTEGER,
    latency_ms      INTEGER,
    user_feedback   SMALLINT CHECK (user_feedback IN (-1, 0, 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
--  AUDITORIA: Log de todas as operações
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    entity      VARCHAR(100) NOT NULL,
    entity_id   UUID,
    action      VARCHAR(50) NOT NULL,
    actor       VARCHAR(100) DEFAULT 'system',
    details     JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
--  ÍNDICES
-- ─────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_data_files_layer    ON data_files (layer, status);
CREATE INDEX IF NOT EXISTS idx_data_files_dataset  ON data_files (dataset_id);
CREATE INDEX IF NOT EXISTS idx_documents_file      ON documents (data_file_id);
CREATE INDEX IF NOT EXISTS idx_documents_metadata  ON documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_rag_runs_created    ON rag_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity    ON audit_log (entity, entity_id);

-- ─────────────────────────────────────────────────────────────
--  TRIGGER: updated_at automático
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_rag_datasets_updated
    BEFORE UPDATE ON rag_datasets
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
