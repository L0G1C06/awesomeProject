# 🔍 RAG Enterprise Platform — ArXiv Dataset

> Plataforma completa de Retrieval-Augmented Generation com Governança Medallion, totalmente local e containerizada.
> **Dataset**: ArXiv (https://info.arxiv.org/help/api/index.html)

---

## 👥 Equipe

| Nome | Email | Matrícula | Papel |
|------|-------|-----------|-------|
| **Eduardo Weber Maldaner** | eduwmaldaner@gmail.com | 211948 | Product Owner (PO) |
| **Lucas Carmargo Oliveira** | Lucaslco2005@gmail.com | 222231 | Scrum Developer |
| **Jeferson Oliveira Moreira** | jef.moreira1@gmail.com | 212148 | Scrum Developer |

### 📋 Informações do Projeto

- **Turma**: CP901TAN1
- **Product Owner**: Eduardo Weber Maldaner
- **Dataset Source**: ArXiv Public API
- **Objetivo**: Plataforma RAG para consulta e análise de artigos científicos do ArXiv

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
| 🥉 **Bronze** | `bronze/` | Dados brutos — artigos ArXiv em `.jsonl` sem transformação |
| 🥈 **Silver** | `silver/` | Dados limpos e normalizados — abstracts e metadados estruturados |
| 🥇 **Gold**   | `gold/`   | Chunks prontos para embedding — seções de artigos segmentadas |

---

## 📚 Backlog Inicial

### Épico 1: Definição de Escopo e Requisitos

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **BS-1** | Escolha do Domínio | Definir foco de pesquisa no ArXiv (ex: Machine Learning, Computer Vision, NLP) | Eduardo | 📋 Backlog |
| **BS-2** | Definição da Empresa Fictícia | Criar contexto de negócio para a plataforma (ex: "TechInsights AI Research") | Jeferson | 📋 Backlog |
| **BS-3** | Problema de Negócio | Documentar o problema que a plataforma RAG resolve para a empresa | Eduardo | 📋 Backlog |
| **BS-4** | Levantamento de Requisitos Funcionais | Mapear features necessárias (busca vetorial, filtros, exportação) | Lucas | 📋 Backlog |
| **BS-5** | Levantamento de Requisitos Não-Funcionais | Definir SLAs, performance, escalabilidade e segurança | Jeferson | 📋 Backlog |
| **BS-6** | Definição de Papéis Scrum | Alinhar responsabilidades: Scrum Master, Product Owner, Developers | Eduardo | 📋 Backlog |

### Épico 2: Integração com ArXiv API

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **AX-1** | Estudo da ArXiv API | Documentar endpoints, limites de taxa e formato de dados | Jeferson | 📋 Backlog |
| **AX-2** | Implementar Connector ArXiv | Criar módulo de conexão com a API | Jeferson | 📋 Backlog |
| **AX-3** | ETL Bronze → Silver (ArXiv) | Normalizar metadata e abstracts dos artigos | Lucas | 📋 Backlog |
| **AX-4** | ETL Silver → Gold | Segmentar artigos em chunks otimizados | Lucas | 📋 Backlog |

### Épico 3: Implementação RAG

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **RAG-1** | Embeddings + Indexação Milvus | Vetorizar chunks e indexar em Milvus | Lucas | 📋 Backlog |
| **RAG-2** | Busca Vetorial | Implementar recuperação top-k com COSINE similarity | Lucas | 📋 Backlog |
| **RAG-3** | Geração com LLM | Construir prompts e gerar respostas via Ollama | Eduardo | 📋 Backlog |
| **RAG-4** | Prompt Engineering | Otimizar templates de prompt para contexto científico | Eduardo | 📋 Backlog |

### Épico 4: Interface e Experiência

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **UI-1** | Frontend Gradio | Criar interface de consulta | Jeferson | 📋 Backlog |
| **UI-2** | Filtros Avançados | Permitir filtrar por categoria, data, autor do ArXiv | Jeferson | 📋 Backlog |
| **UI-3** | Exibição de Resultados | Mostrar snippets com highlightning de trechos | Lucas | 📋 Backlog |
| **UI-4** | Exportação de Resultados | Gerar relatórios em PDF/CSV | Jeferson | 📋 Backlog |

### Épico 5: Observabilidade e Monitoramento

| ID | Tarefa | Descrição | Responsável | Status |
|----|--------|-----------|-------------|--------|
| **OBS-1** | MLflow Tracking | Registrar queries, latência e qualidade de respostas | Lucas | 📋 Backlog |
| **OBS-2** | Dashboard de Performance | Criar dashboard com métricas de uso | Eduardo | 📋 Backlog |
| **OBS-3** | Alertas e Logs | Configurar logs estruturados e alertas | Jeferson | 📋 Backlog |

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
│   ├── ingestion/ingest.py     # Ingestão do arXiv + persistência local/MinIO
│   ├── processing/
│   │   ├── bronze_to_silver.py # Limpeza e normalização do schema do arXiv
│   │   └── silver_to_gold.py   # Chunking para indexação
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

## 🔧 Configurando seu Dataset - ArXiv

### 📋 Dataset: ArXiv Open Access

Este projeto utiliza a **ArXiv Public API** para recuperar artigos científicos.

**Documentação**: https://info.arxiv.org/help/api/index.html

**Categorias disponíveis no ArXiv**:
- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.CV` — Computer Vision
- `cs.NLP` — Natural Language Processing
- `cs.CL` — Computation and Language
- `physics.data-an` — Data Analysis
- `stat.ML` — Statistics Machine Learning
- E muitas outras...

### Passo 1: Defina a categoria de pesquisa
Edite `.env` e configure:
```bash
ARXIV_CATEGORY="cs.LG"  # Exemplo: Machine Learning
ARXIV_MAX_RESULTS=10000 # Quantidade de artigos para ingestão
ARXIV_BATCH_SIZE=2000   # Máximo por página da API
ARXIV_DELAY_SECONDS=3   # Recomendado pelo arXiv para múltiplas chamadas
DATASET_DOMAIN="Machine Learning Research"
DATASET_SOURCE_URL="https://export.arxiv.org/api/query"
```

### Passo 2: Baixe 10.000 amostras localmente
Sem depender da infraestrutura completa, você pode salvar a amostra em disco:
```bash
python -m pipeline.ingestion.ingest \
  --categories cs.LG \
  --max-results 10000 \
  --output-dir data/bronze \
  --skip-minio
```

O arquivo JSONL será salvo em `data/bronze/arxiv/raw/`.

### Passo 3: Explore no notebook
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### Passo 4: Ajustes restantes
Arquivos que ainda podem ser refinados conforme o domínio do projeto:

1. **`api/services/rag_service.py`** — Ajustar prompt para contexto científico
2. **`.env`** — Refinar categorias, volume de ingestão e ordenação da coleta
3. **`pipeline/embedding/embed_and_index.py`** — Ajustar estratégia de indexação conforme o volume de chunks

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

## 🎯 Papéis Scrum

| Papel | Descrição | Responsável |
|-------|-----------|-------------|
| **Product Owner (PO)** | Define requisitos, prioriza backlog, valida entregas | Eduardo Weber Maldaner |
| **Scrum Master** | Facilita cerimônias, remove impedimentos, protege o time | Eduardo Weber Maldaner |
| **Developer** | Implementa features, garante qualidade, autoorganizado | Lucas Carmargo, Jeferson |

### Cerimônias

- **Sprint**: 1 semana
- **Planning**: Segundas (10:00) — Define sprint backlog
- **Daily**: Terça-Sexta (09:00) — Sincronização rápida
- **Review**: Segundas fim de sprint (14:00) — Demonstra entregáveis
- **Retrospectiva**: Segundas fim de sprint (15:00) — Melhoria contínua

### Objetivo do Sprint 1

✅ Completar épico **BS** (Definição de Escopo e Requisitos)
✅ Iniciar integração com ArXiv API (**AX-1**, **AX-2**)

---

## 📚 Referências

- [ArXiv API Documentation](https://info.arxiv.org/help/api/index.html)
- [Ollama Models](https://ollama.com/library)
- [Milvus Docs](https://milvus.io/docs)
- [MLflow Docs](https://mlflow.org/docs/latest)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [MinIO Docs](https://min.io/docs)
