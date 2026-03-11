# 🔍 RAG Enterprise Platform

> Plataforma completa de Retrieval-Augmented Generation com Governança Medallion, totalmente local e containerizada.

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        APLICAÇÃO                                │
│   Gradio :7860  ←→  FastAPI :8000                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       CAMADA DE IA                               │
│   Ollama (LLM + Embeddings)   ←→   MLflow (Tracking)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    CAMADA DE DADOS                               │
│   MinIO (Bronze/Silver/Gold)  Milvus (Vetorial)  PostgreSQL     │
└─────────────────────────────────────────────────────────────────┘
```

### Governança Medallion

| Camada | Bucket | Conteúdo |
|--------|--------|----------|
| 🥉 **Bronze** | `bronze/` | Dados brutos — `.jsonl` sem transformação |
| 🥈 **Silver** | `silver/` | Dados limpos e normalizados |
| 🥇 **Gold**   | `gold/`   | Chunks prontos para embedding |

### Ferramentas de Dados

- **PostgreSQL**: metadados do pipeline, controle de versionamento por dataset e auditoria de eventos.
- **Milvus**: armazenamento de embeddings e indexação vetorial para retrieval semântico.

---

## 🚀 Quick Start

### Pré-requisitos
- Docker >= 24.0
- Docker Compose >= 2.20
- Make
- 16GB RAM recomendado (para LLM local)
- GPU opcional (melhora performance do Ollama)

### 1. Clone e configure
```bash
git clone <repo-url>
cd rag-enterprise
make env          # Cria .env a partir do .env.example
```

### 2. Suba a infraestrutura
```bash
make up           # Sobe todos os serviços
make pull-models  # Baixa modelos LLM e embedding
```

### 3. Execute o pipeline
```bash
# Antes: implemente load_raw_data() em pipeline/ingestion/ingest.py
make pipeline     # ingest → process → embed
```

### 4. Acesse os serviços

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **Frontend (Gradio)** | http://localhost:7860 | — |
| **API (FastAPI)** | http://localhost:8000/docs | — |
| **MLflow** | http://localhost:5000 | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **PostgreSQL** | localhost:5432 | raguser / ragpass |

---

## 📁 Estrutura do Projeto

```
rag-enterprise/
├── 📄 docker-compose.yml       # Orquestração de todos os serviços
├── 📄 Makefile                 # Automação de tarefas
├── 📄 .env.example             # Template de variáveis de ambiente
│
├── 📂 api/                     # FastAPI
│   ├── main.py
│   ├── core/config.py          # Configurações centralizadas
│   ├── routers/                # Endpoints HTTP
│   ├── schemas/                # Modelos Pydantic
│   ├── services/               # Lógica de negócio
│   └── middleware/
│
├── 📂 pipeline/                # Pipeline RAG
│   ├── ingestion/ingest.py     # ← ADAPTAR: carregamento do dataset
│   ├── processing/
│   │   ├── bronze_to_silver.py # ← ADAPTAR: limpeza de dados
│   │   └── silver_to_gold.py   # ← ADAPTAR: chunking
│   └── embedding/
│       └── embed_and_index.py  # Embeddings + Milvus
│
├── 📂 frontend/                # Interface Gradio
│   └── app.py
│
├── 📂 infra/                   # Configurações de infraestrutura
│   ├── postgres/init.sql       # Schema inicial do banco
│   └── ollama/entrypoint.sh    # Auto-download de modelos
│
├── 📂 tests/                   # Testes
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── 📂 notebooks/               # Exploração de dados
│   └── 01_dataset_exploration.ipynb
│
└── 📂 docs/
    └── diagrams/architecture.md
```

---

## 🔧 Configurando seu Dataset

### Passo 1: Escolha o dataset
Consulte as fontes sugeridas:
- https://github.com/awesomedata/awesome-public-datasets
- https://github.com/petrobras/3W
- https://github.com/jonathanwvd/awesome-industrial-datasets

### Passo 2: Explore no notebook
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### Passo 3: Implemente os TODOs
Arquivos que precisam ser adaptados (marcados com `# TODO`):

1. **`pipeline/ingestion/ingest.py`** — `load_raw_data()`
2. **`pipeline/processing/bronze_to_silver.py`** — `clean_record()`
3. **`pipeline/processing/silver_to_gold.py`** — `extract_text()` e `extract_metadata()`
4. **`api/services/rag_service.py`** — `_build_prompt()`
5. **`.env`** — `DATASET_DOMAIN` e `DATASET_SOURCE_URL`

### Ingestão em volume (Data Lake)

Para puxar mais dados do arXiv sem estourar memória, a ingestão Bronze usa paginação:

- `ARXIV_MAX_RESULTS`: total alvo de registros por execução
- `ARXIV_PAGE_SIZE`: tamanho de cada página/arquivo bruto
- `ARXIV_MAX_PAGES`: limite de páginas (`0` = sem limite)
- `ARXIV_REQUEST_DELAY_SECONDS`: intervalo entre chamadas da API
- `ARXIV_DEDUP_IN_RUN`: remove IDs duplicados dentro da mesma execução
- `ARXIV_RETRY_ATTEMPTS`, `ARXIV_RETRY_BACKOFF_SECONDS`, `ARXIV_RETRY_BACKOFF_FACTOR`: resiliência para erro 429/5xx da API
- `ARXIV_FAIL_FAST`: se `false`, mantém dados já coletados mesmo com falha em uma página

Exemplo de coleta maior:

```bash
ARXIV_MAX_RESULTS=20000 ARXIV_PAGE_SIZE=200 make ingest
```

---

## 📋 Comandos Make

```bash
make help           # Lista todos os comandos
make up             # Sobe todos os serviços
make down           # Para serviços
make restart        # Reinicia
make logs           # Logs de todos os serviços
make logs-api       # Logs de um serviço específico
make status         # Status dos containers

make pull-models    # Baixa modelos Ollama
make pipeline       # Executa pipeline completo
make ingest         # Apenas ingestão Bronze
make process        # Bronze → Silver → Gold
make embed          # Embedding + indexação Milvus

make db-migrate     # Aplica migrações PostgreSQL
make db-shell       # Shell interativo do banco

make test           # Todos os testes
make test-unit      # Testes unitários
make lint           # Linter (ruff)
make format         # Formatter (black + ruff)

make open-all       # Lista URLs de todos os serviços
make clean          # Remove containers e volumes
```

---

## 🔄 Fluxo do Pipeline RAG

```
Query do usuário
     │
     ▼
Embedding da query (Ollama: nomic-embed-text)
     │
     ▼
Busca vetorial (Milvus: top-k por COSINE similarity)
     │
     ▼
Construção do prompt (context + query)
     │
     ▼
Geração (Ollama: llama3.2)
     │
     ▼
Log MLflow + Persistência PostgreSQL
     │
     ▼
Resposta ao usuário
```

---

## 📊 MLOps

Cada query RAG é automaticamente registrada no MLflow com:
- Parâmetros: modelo LLM, embed model, top-k, query
- Métricas: latência, número de docs recuperados
- Artefatos: prompt usado, resposta gerada

Acesse: http://localhost:5000

---

## 🧪 Testes

```bash
make test           # Todos os testes com coverage
make test-unit      # Apenas unitários (sem infra necessária)
make test-integration  # Requer serviços rodando
```

---

## 🐛 Troubleshooting

**Ollama sem GPU**: Remova o bloco `deploy.resources` do `docker-compose.yml` na seção `ollama`.

**Milvus não sobe**: Verifique se o etcd e minio estão saudáveis primeiro com `make status`.

**Modelos lentos**: Use modelos menores como `phi3` ou `tinyllama` no `.env`.

---

## 📚 Referências

- [Ollama Models](https://ollama.com/library)
- [Milvus Docs](https://milvus.io/docs)
- [MLflow Docs](https://mlflow.org/docs/latest)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [MinIO Docs](https://min.io/docs)
