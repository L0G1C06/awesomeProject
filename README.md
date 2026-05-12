# 🔍 RAG Enterprise Platform — ArXiv Dataset

> Plataforma completa de Retrieval-Augmented Generation com Governança Medallion, executada localmente via Docker e com suporte a múltiplos provedores de LLM (Ollama, HuggingFace e OpenAI).
> **Dataset**: ArXiv ([https://info.arxiv.org/help/api/index.html](https://info.arxiv.org/help/api/index.html))

---

## 👥 Equipe

| Nome | Email | Matrícula | Papel |
|------|-------|-----------|-------|
| **Eduardo Weber Maldaner** | eduwmaldaner@gmail.com | 211948 | Product Owner (PO) |
| **Lucas Carmargo Oliveira** | lucaslco2005@gmail.com | 222231 | Scrum Developer |
| **Jeferson Oliveira Moreira** | jef.moreira1@gmail.com | 212148 | Scrum Developer |
| **Wallace Eron Melo de Barros** | wallaceerom7@gmail.com | 211751 | Scrum Developer |
| **Heifor Barreto** | heiforbarreto@gmail.com | 224541 | Scrum Developer |
| **Nicola Luca Tognocchi** | nicolatognocchi33x@gmail.com | 223138 | Scrum Developer |
| **Arthur Soares Maffeis** | arthurmaffeis@hotmail.com | 150448 | Scrum Developer |

### 📋 Informações do Projeto

- **Turma**: CP901TAN1
- **Product Owner**: Eduardo Weber Maldaner
- **Dataset Source**: ArXiv Public API
- **Objetivo**: Plataforma RAG para consulta e análise de artigos científicos do ArXiv

---

## 🏗️ Arquitetura

```
┌──────────────────────────────────────────────────────────────────────┐
│                           APLICAÇÃO                                  │
│   Gradio :7860  ──→  FastAPI :8000  (/query/, /docs)                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│                          CAMADA DE IA                                │
│   Provedores LLM:                                                    │
│     • Ollama (local)        → embeddings + LLM                       │
│     • HuggingFace (remoto)  → embeddings + LLM + reranker (local)    │
│     • OpenAI (remoto)       → LLM (embeddings via HuggingFace)       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│                        CAMADA DE DADOS                               │
│   MinIO  (Bronze/Silver/Gold)   Milvus  (Vetorial)   PostgreSQL      │
│                                  ↑                                   │
│                           Attu :8080  (UI Milvus)                    │
└──────────────────────────────────────────────────────────────────────┘
```

> Diagrama detalhado em [docs/arquitetura.md](docs/arquitetura.md).

### Governança Medallion

| Camada | Bucket | Conteúdo |
|--------|--------|----------|
| 🥉 **Bronze** | `bronze/` | Dados brutos — artigos ArXiv em `.jsonl` sem transformação |
| 🥈 **Silver** | `silver/` | Dados limpos e normalizados — abstracts e metadados estruturados |
| 🥇 **Gold**   | `gold/`   | Chunks prontos para embedding — seções de artigos segmentadas |

### Provedores de LLM

A escolha do provider é dinâmica, podendo vir do payload da requisição (`llm_provider` / `llm_model`), do auto-detect pelo nome do modelo ou da variável `LLM_PROVIDER` no `.env`.

| Provider | Embedding | LLM | Reranker | Observações |
|----------|-----------|-----|----------|-------------|
| **`ollama`** | `nomic-embed-text` (local) | `OLLAMA_LLM_MODEL` (ex.: `mistral`, `llama3.2`) | — | 100% local; baixa modelos automaticamente no startup |
| **`huggingface`** | `BAAI/bge-base-en-v1.5` (Inference API) | `HF_LLM_MODEL` (ex.: `meta-llama/Meta-Llama-3-8B-Instruct`) | `CrossEncoder` local em `models/reranker/` (top-k → top-3) | Requer `HUGGINGFACE_API_TOKEN` |
| **`openai`** | `BAAI/bge-base-en-v1.5` (HuggingFace) | `OPENAI_MODEL` (ex.: `gpt-4o-mini`) | — | Requer `OPENAI_API_KEY`; embeddings continuam vindo do HF para preservar compatibilidade com o índice já existente |

Auto-detect (em `api/services/rag_service.py::_detect_provider`):
- Modelo começando com `gpt-` ou `gpt4` → **openai**
- Modelo no formato `org/model` (com `/`) → **huggingface**
- Demais nomes simples → **ollama**

---

## 📚 Backlog Inicial

### Épico 1: Definição de Escopo e Requisitos

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **BS-1** | Escolha do Domínio | Definir foco de pesquisa no ArXiv (ex: Machine Learning, Computer Vision, NLP) | Eduardo | ✅ Concluído |
| **BS-2** | Definição da Empresa Fictícia | Criar contexto de negócio para a plataforma | Jeferson | ✅ Concluído |
| **BS-3** | Problema de Negócio | Documentar o problema que a plataforma RAG resolve para a empresa | Eduardo | ✅ Concluído |
| **BS-4** | Levantamento de Requisitos Funcionais | Mapear features necessárias (busca vetorial, filtros, exportação) | Lucas | ✅ Concluído |
| **BS-5** | Levantamento de Requisitos Não-Funcionais | Definir SLAs, performance, escalabilidade e segurança | Jeferson | ✅ Concluído |
| **BS-6** | Definição de Papéis Scrum | Alinhar responsabilidades: Scrum Master, Product Owner, Developers | Eduardo | ✅ Concluído |

### Épico 2: Integração com ArXiv API

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **AX-1** | Estudo da ArXiv API | Documentar endpoints, limites de taxa e formato de dados | Jeferson | ✅ Concluído |
| **AX-2** | Implementar Connector ArXiv | Criar módulo de conexão com a API | Jeferson | ✅ Concluído |
| **AX-3** | ETL Bronze → Silver (ArXiv) | Normalizar metadata e abstracts dos artigos | Lucas | ✅ Concluído |
| **AX-4** | ETL Silver → Gold | Segmentar artigos em chunks otimizados | Lucas | ✅ Concluído |

### Épico 3: Implementação RAG

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **RAG-1** | Embeddings + Indexação Milvus | Vetorizar chunks e indexar em Milvus (HNSW + COSINE) | Lucas | ✅ Concluído |
| **RAG-2** | Busca Vetorial | Implementar recuperação top-k por similaridade COSINE | Lucas | ✅ Concluído |
| **RAG-3** | Geração com LLM | Construir prompts e gerar respostas via Ollama / HuggingFace / OpenAI | Eduardo | ✅ Concluído |
| **RAG-4** | Prompt Engineering | Otimizar templates de prompt para contexto científico | Eduardo | ✅ Concluído |
| **RAG-5** | Re-ranker (CrossEncoder) | Reordenar documentos recuperados via modelo treinado localmente | Lucas | ✅ Concluído |

### Épico 4: Interface e Experiência

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **UI-1** | Frontend Gradio | Criar interface de consulta | Jeferson | ✅ Concluído |
| **UI-2** | Seletor de Provider/Modelo | Permitir escolher entre Ollama / HuggingFace / OpenAI | Jeferson | ✅ Concluído |
| **UI-3** | Exibição de Resultados | Mostrar snippets, scores, autores, categorias e links arXiv | Lucas | ✅ Concluído |
| **UI-4** | Filtros Avançados | Filtrar por categoria, data, autor | Jeferson | ✅ Concluído |
| **UI-5** | Exportação de Resultados | Gerar relatórios em PDF/CSV | Jeferson | ✅ Concluído |

### Épico 5: Observabilidade e Monitoramento

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **OBS-1** | MLflow Tracking | Registrar queries, latência, tokens e artefatos | Lucas | 📋 Backlog |
| **OBS-2** | Persistência em PostgreSQL | Auditoria, runs RAG, versionamento de datasets | Jeferson | ✅ Concluído |
| **OBS-3** | Dashboard de Performance | Criar dashboard com métricas de uso | Eduardo | 📋 Backlog |
| **OBS-4** | Alertas e Logs | Configurar logs estruturados e alertas | Jeferson | 📋 Backlog |

### Ferramentas de Dados

- **PostgreSQL**: metadados do pipeline, controle de versionamento por dataset, registro de runs RAG e auditoria de eventos.
- **Milvus**: armazenamento de embeddings e indexação vetorial (HNSW / COSINE) para retrieval semântico.
- **MinIO**: data lake S3-compatible com camadas Bronze, Silver e Gold.
- **Attu**: console visual do Milvus disponível em `http://localhost:8080`.

---

## 🚀 Quick Start

### Pré-requisitos
- Docker >= 24.0
- Docker Compose >= 2.20
- Make
- 16GB RAM recomendado (para LLM local via Ollama)
- GPU opcional (acelera Ollama e o reranker)

### 1. Clone e configure

```bash
git clone <repo-url>
cd awesomeProject
```

Crie o arquivo `.env` na raiz do projeto. Veja o exemplo completo em [Variáveis de Ambiente](#-variáveis-de-ambiente).

### 2. Suba a infraestrutura

```bash
make up           # Sobe todos os serviços (build + up -d)
```

> O container do Ollama executa `infra/ollama/entrypoint.sh`, que **baixa automaticamente** os modelos definidos em `OLLAMA_LLM_MODEL` e `OLLAMA_EMBED_MODEL` na primeira inicialização. O alvo `make pull-models` continua disponível para forçar o download de modelos extras.

### 3. Execute o pipeline

```bash
make pipeline     # ingest → process → embed
```

Etapas individuais:

```bash
make ingest       # arXiv API → bucket Bronze
make process      # Bronze → Silver → Gold
make embed        # Embeddings + indexação no Milvus
```

### 4. Acesse os serviços

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **Frontend (Gradio)** | http://localhost:7860 | — |
| **API (FastAPI)** | http://localhost:8000/docs | — |
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin` |
| **Attu (Milvus UI)** | http://localhost:8080 | — |
| **PostgreSQL** | `localhost:5433` | `raguser` / `ragpass` (db: `ragdb`) |
| **Milvus (gRPC)** | `localhost:19530` | — |
| **Ollama** | http://localhost:11435 | — |

### 5. Selecionando o provider de LLM

Existem três formas, em ordem de prioridade:

1. **Por requisição** — campo `llm_provider` (e opcionalmente `llm_model`) no payload de `POST /query/`.
2. **Auto-detect** — se apenas `llm_model` for enviado, o provider é inferido pelo nome.
3. **Padrão global** — variável `LLM_PROVIDER` do `.env` (`ollama`, `huggingface` ou `openai`).

Exemplo via OpenAI:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=<sua-chave>
OPENAI_MODEL=gpt-4o-mini
OPENAI_REASONING_EFFORT=low
OPENAI_TIMEOUT_SECONDS=120
```

> Mesmo no fluxo OpenAI os embeddings continuam sendo gerados pelo HuggingFace (`BAAI/bge-base-en-v1.5`) para preservar compatibilidade com o índice vetorial já existente.

---

## 📁 Estrutura do Projeto

```
awesomeProject/
├── 📄 docker-compose.yml          # Orquestração de todos os serviços
├── 📄 Makefile                    # Automação de tarefas
├── 📄 .env                        # Variáveis de ambiente (NÃO commitar)
├── 📄 requirements.txt            # Dependências Python (ambiente local/dev)
│
├── 📂 api/                        # Backend FastAPI
│   ├── Dockerfile
│   ├── main.py                    # Bootstrap FastAPI + CORS + routers
│   ├── requirements.runtime.txt   # Deps mínimas para o container
│   ├── core/                      # (placeholder)
│   ├── routers/
│   │   └── query.py               # POST /query/  (rota principal RAG)
│   ├── schemas/
│   │   ├── config.py              # Settings (pydantic-settings) lendo .env
│   │   └── query.py               # QueryRequest / QueryResponse / RetrievedDoc
│   └── services/
│       ├── rag_service.py         # Orquestrador (roteamento dos providers)
│       ├── rag_service_ollama.py  # Implementação Ollama (legado/standalone)
│       ├── rag_huggingface_service.py # Cliente HF Inference API (embed + chat)
│       ├── openai_service.py      # Cliente OpenAI (Chat Completions)
│       ├── reranker_service.py    # CrossEncoder local (models/reranker/)
│       ├── milvus_service.py      # Coleção, índice HNSW e busca vetorial
│       └── postgres_service.py    # Datasets, versões, runs e auditoria
│
├── 📂 pipeline/
│   ├── ingestion/
│   │   └── ingest.py              # arXiv API → Bronze (local + MinIO)
│   ├── processing/
│   │   ├── bronze_to_silver.py    # Limpeza e normalização do schema arXiv
│   │   └── silver_to_gold.py      # Chunking (RecursiveCharacterTextSplitter)
│   ├── embedding/
│   │   └── embed_and_index.py     # SentenceTransformer + Milvus (idempotente)
│   └── reranker/
│       ├── generate_reranker_dataset.py
│       ├── train_reranker.py
│       └── upload_reranker_model.py
│
├── 📂 frontend/                   # Interface Gradio
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.runtime.txt
│
├── 📂 infra/
│   ├── postgres/init.sql          # Schema inicial (datasets, runs, audit_log…)
│   └── ollama/entrypoint.sh       # Auto-download dos modelos no startup
│
├── 📂 models/
│   └── reranker/                  # CrossEncoder local (config + safetensors)
│
├── 📂 data/                       # Saídas locais do pipeline (Bronze/Silver)
│   ├── bronze/arxiv/raw/
│   └── silver/exploration/
│
├── 📂 tests/
│   └── test_pipeline.py
│
├── 📂 notebooks/
│   └── 01_dataset_exploration.ipynb
│
└── 📂 docs/
    └── arquitetura.md             # Diagrama Mermaid completo
```

---

## 🔧 Variáveis de Ambiente

Todas configuradas via `.env` na raiz do projeto. Os defaults usados em containers vêm do `docker-compose.yml`.

### MinIO / S3

```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
BUCKET_BRONZE=bronze
BUCKET_SILVER=silver
BUCKET_GOLD=gold
```

### PostgreSQL

```bash
DATABASE_URL=postgresql://raguser:ragpass@localhost:5433/ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_DB=ragdb
```

### Milvus

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_documents
```

### Ollama (LLM local)

```bash
OLLAMA_HOST=http://localhost:11435      # host externo (mapeado para 11434 interno)
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EMBED_DIMENSION=768
```

### HuggingFace

```bash
HUGGINGFACE_API_TOKEN=hf_xxx
HF_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HF_EMBED_MODEL=BAAI/bge-base-en-v1.5
```

### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_REASONING_EFFORT=low             # opcional, modelos GPT-5/o1
OPENAI_TIMEOUT_SECONDS=120
```

### Provider padrão

```bash
LLM_PROVIDER=ollama                     # ollama | huggingface | openai
```

### Pipeline RAG

```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=5
EMBED_MODEL=BAAI/bge-base-en-v1.5       # usado por pipeline/embedding/embed_and_index.py
BATCH_SIZE=64
```

### Dataset (arXiv)

```bash
DATASET_NAME=arxiv
DATASET_DOMAIN=research
DATASET_SOURCE_URL=https://arxiv.org/api/query
DATASET_VERSION=1.0.0
```

### Ingestão arXiv

Variáveis lidas por [pipeline/ingestion/ingest.py](pipeline/ingestion/ingest.py):

| Variável | Default | Função |
|----------|---------|--------|
| `ARXIV_CATEGORY` | `cs.LG` | Categorias (separadas por vírgula) — combinadas com OR |
| `ARXIV_MAX_RESULTS` | `10000` | Total de artigos a baixar por execução |
| `ARXIV_BATCH_SIZE` | `2000` | Tamanho da página da API arXiv (máx. recomendado) |
| `ARXIV_DELAY_SECONDS` | `3` | Delay entre páginas (recomendado pelo arXiv) |
| `ARXIV_SORT_BY` | `submittedDate` | Critério de ordenação |
| `ARXIV_SORT_ORDER` | `descending` | Ordem |
| `ARXIV_OUTPUT_DIR` | `data/bronze` | Pasta local de saída JSONL |
| `ARXIV_DATASET_NAME` | `arxiv` | Nome lógico em `rag_datasets` |
| `ARXIV_WRITE_TO_MINIO` | `true` | Se `false`, salva apenas no disco local |
| `ARXIV_USER_AGENT` | `awesomeProject/1.0` | User-Agent enviado ao arXiv |

### Frontend (Gradio)

```bash
API_URL=http://localhost:8000
GRADIO_SERVER_PORT=7860
TOP_K_RETRIEVAL=5
```

> Dentro do `docker-compose`, `API_URL` é sobrescrito para `http://api:8000` automaticamente.

### Exportação de Resultados

A exportação (UI-5) é feita diretamente no frontend após cada consulta. Dois botões ficam disponíveis abaixo dos resultados:

| Formato | Biblioteca | Colunas / Conteúdo |
|---------|-----------|-------------------|
| **CSV** | stdlib `csv` | `rank`, `score`, `title`, `authors`, `categories`, `primary_category`, `published`, `arxiv_id`, `url`, `content` |
| **PDF** | `fpdf2` | Consulta, resposta e lista de documentos com metadados e excerpt |

---

## 🔧 Configurando seu Dataset — ArXiv

### 📋 Dataset: ArXiv Open Access

Este projeto utiliza a **ArXiv Public API** para recuperar artigos científicos.

**Documentação**: https://info.arxiv.org/help/api/index.html

**Categorias mais usadas**:

- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.CV` — Computer Vision
- `cs.CL` — Computation and Language (NLP)
- `q-bio.GN` — Genomics
- `physics.data-an` — Data Analysis
- `stat.ML` — Statistics / Machine Learning

### Passo 1: Defina a categoria de pesquisa

Edite `.env`:

```bash
ARXIV_CATEGORY="cs.LG"
ARXIV_MAX_RESULTS=10000
ARXIV_BATCH_SIZE=2000
ARXIV_DELAY_SECONDS=3
DATASET_DOMAIN="Machine Learning Research"
```

> O arquivo [pipeline/ingestion/ingest.py](pipeline/ingestion/ingest.py) também aceita uma lista interna `CATEGORY_BATCHES` para coletar lotes independentes de categorias em uma mesma execução (ex.: `cs.LG` + `q-bio.GN`).

### Passo 2: Ingestão local sem subir a infra

```bash
python -m pipeline.ingestion.ingest \
  --categories cs.LG \
  --max-results 10000 \
  --output-dir data/bronze \
  --no-write-to-minio
```

O JSONL é salvo em `data/bronze/arxiv/raw/`.

### Passo 3: Explore no notebook

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### Passo 4: Ajustes finos

- **`api/services/rag_service.py`** — `SYSTEM_PROMPT` e `_build_prompt` para contexto científico.
- **`pipeline/processing/silver_to_gold.py`** — `CHUNK_SIZE` / `CHUNK_OVERLAP` para granularidade dos chunks.
- **`pipeline/embedding/embed_and_index.py`** — `EMBED_MODEL` e `BATCH_SIZE` conforme volume/hardware.

---

## 📋 Comandos Make

```bash
make help            # Lista todos os comandos
make up              # Sobe todos os serviços
make down            # Para serviços
make restart         # Reinicia
make build           # Rebuild sem cache
make logs            # Logs de todos os serviços
make logs-api        # Logs do serviço api (use logs-<serviço>)
make status          # Status dos containers
make infra-up        # Sobe apenas infraestrutura (sem api/frontend)
make infra-down      # Para apenas infraestrutura

make pull-models     # Força download de modelos Ollama (llama3.2 + nomic-embed-text)
make list-models     # Lista modelos disponíveis no container Ollama

make pipeline        # Executa pipeline completo (ingest → process → embed)
make ingest          # Apenas ingestão Bronze
make process         # Bronze → Silver → Gold
make embed           # Embedding + indexação Milvus

make db-migrate      # alembic upgrade head (caso existam migrations)
make db-rollback     # alembic downgrade -1
make db-shell        # Shell interativo do PostgreSQL

make test            # Todos os testes
make test-unit       # Testes unitários
make test-integration
make test-e2e
make test-cov        # Testes com cobertura HTML

make lint            # Linter (ruff)
make format          # Formatter (ruff format + black)

make open-all        # Lista URLs de todos os serviços (API, Gradio, MinIO, Attu, Milvus, PostgreSQL)
make clean           # Remove containers, volumes e imagens órfãs
```

> ⚠️ O alvo `make env` espera um `.env.example`. Caso não exista, crie o `.env` manualmente (ver [Variáveis de Ambiente](#-variáveis-de-ambiente)).

---

## 🔄 Fluxo do Pipeline RAG

### Pipeline de dados (offline)

```
arXiv API
   │  pipeline/ingestion/ingest.py
   ▼
Bronze (MinIO + data/bronze/)        — JSONL bruto
   │  pipeline/processing/bronze_to_silver.py
   ▼
Silver                               — abstracts limpos + metadados
   │  pipeline/processing/silver_to_gold.py
   ▼
Gold                                 — chunks (CHUNK_SIZE/OVERLAP)
   │  pipeline/embedding/embed_and_index.py
   ▼
Milvus (rag_documents, HNSW/COSINE) + PostgreSQL (data_files, documents)
```

### Pipeline de query (online)

```
Query do usuário (Gradio → POST /query/)
   │
   ▼
Embedding da query
   ├── ollama       → nomic-embed-text (local)
   └── hf / openai  → BAAI/bge-base-en-v1.5 (HF Inference)
   │
   ▼
Busca vetorial (Milvus, top-k por COSINE)
   │
   ▼
[somente HF]  Reranker CrossEncoder local (top-k → top-3)
   │
   ▼
Construção do prompt (SYSTEM_PROMPT + contexto + query)
   │
   ▼
Geração
   ├── ollama       → ollama.Client.chat (OLLAMA_LLM_MODEL)
   ├── huggingface  → InferenceClient.chat_completion (HF_LLM_MODEL)
   └── openai       → OpenAI Chat Completions (OPENAI_MODEL)
   │
   ▼
Persistência PostgreSQL (rag_runs + audit_log)
   │
   ▼
QueryResponse (answer, retrieved_docs, latency_ms, run_id)
```

---

## 🗃️ Schema PostgreSQL

Inicializado automaticamente por [infra/postgres/init.sql](infra/postgres/init.sql).

| Tabela | Função |
|--------|--------|
| `rag_datasets` | Datasets lógicos (nome, domínio, source URL, versão) |
| `rag_dataset_versions` | Histórico de versões por dataset |
| `data_files` | Rastreio de arquivos por camada Medallion (bronze/silver/gold) com `status` e `checksum` |
| `documents` | Vínculo chunk ↔ embedding (`milvus_id`, `embed_model`, `metadata`) |
| `rag_runs` | Cada execução de query RAG (query, prompt, resposta, latência, top-k, feedback) |
| `audit_log` | Eventos auditáveis em formato JSONB |

---

## 🌐 API HTTP

### `POST /query/`

```jsonc
// Request
{
  "query": "What are recent advances in retrieval-augmented generation?",
  "top_k": 5,
  "llm_model": "gpt-4o-mini",        // opcional
  "llm_provider": "openai",           // opcional — ollama | huggingface | openai
  "filter_category": "cs.LG",        // opcional — categoria arXiv (ex: cs.LG, cs.CV)
  "filter_author": "Hinton",          // opcional — parte do nome do autor
  "filter_date_from": "2023-01-01",  // opcional — YYYY-MM-DD
  "filter_date_to": "2024-12-31"     // opcional — YYYY-MM-DD
}
```

```jsonc
// Response (QueryResponse)
{
  "run_id": "…",
  "query": "…",
  "answer": "…",
  "retrieved_docs": [
    {
      "score": 0.81,
      "title": "…",
      "url": "…",
      "arxiv_id": "2401.01234",
      "authors": "…",
      "categories": "cs.CL cs.AI",
      "primary_category": "cs.CL",
      "content": "…",
      "published": "…",
      "updated": "…"
    }
  ],
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "latency_ms": 2840
}
```

Documentação interativa em http://localhost:8000/docs.

---

## 🧪 Testes

```bash
make test              # Todos os testes
make test-unit         # Apenas unitários
make test-integration  # Requer infra rodando
make test-cov          # Cobertura HTML
```

> Atualmente o repositório contém [tests/test_pipeline.py](tests/test_pipeline.py); as estruturas `tests/unit`, `tests/integration` e `tests/e2e` estão previstas no Makefile mas ainda não populadas.

---

## 🐛 Troubleshooting

- **Ollama sem GPU**: o `docker-compose.yml` atual já não declara `deploy.resources` para o Ollama — ele roda em CPU por padrão. Para acelerar, adicione um bloco `deploy.resources.reservations.devices` apontando para a GPU.
- **Milvus não sobe**: verifique se `etcd` e `minio` estão saudáveis primeiro (`make status`). Milvus depende de ambos.
- **Modelos Ollama lentos**: use modelos menores (`phi3`, `tinyllama`, `llama3.2:1b`) ajustando `OLLAMA_LLM_MODEL` no `.env`.
- **`OPENAI_API_KEY não configurada`**: defina a chave no `.env` antes de selecionar `LLM_PROVIDER=openai`.
- **Erro 429 / quota OpenAI**: o endpoint `/query/` traduz para HTTP 429 com mensagem orientando a verificar o billing em https://platform.openai.com/account/billing/overview.
- **Reranker desabilitado**: se `sentence-transformers` não estiver instalado ou `models/reranker/` não existir, o `RerankerService` apenas retorna `docs[:top_k]` sem reordenação — o pipeline continua funcionando normalmente.
- **`make env` falha**: este projeto não distribui `.env.example`; crie o `.env` manualmente com base em [Variáveis de Ambiente](#-variáveis-de-ambiente).

---

## 🎯 Papéis Scrum

| Papel | Descrição | Responsável |
|-------|-----------|-------------|
| **Product Owner (PO)** | Define requisitos, prioriza backlog, valida entregas | Eduardo Weber Maldaner |
| **Scrum Master** | Facilita cerimônias, remove impedimentos, protege o time | Eduardo Weber Maldaner |
| **Developer** | Implementa features, garante qualidade, autoorganizado | Lucas Carmargo, Jeferson, Wallace, Heifor, Nicola, Arthur |

### Cerimônias

- **Sprint**: 1 semana
- **Planning**: Segundas (10:00) — Define sprint backlog
- **Daily**: Terça-Sexta (09:00) — Sincronização rápida
- **Review**: Segundas fim de sprint (14:00) — Demonstra entregáveis
- **Retrospectiva**: Segundas fim de sprint (15:00) — Melhoria contínua

---

## 📚 Referências

- [ArXiv API Documentation](https://info.arxiv.org/help/api/index.html)
- [Ollama Models](https://ollama.com/library)
- [Milvus Docs](https://milvus.io/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [MinIO Docs](https://min.io/docs)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Sentence-Transformers / CrossEncoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
