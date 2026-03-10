# ═══════════════════════════════════════════════════════════════
#  RAG Enterprise Platform — Makefile
#  Uso: make <comando>
# ═══════════════════════════════════════════════════════════════

.PHONY: help up down restart logs clean build \
        infra-up infra-down \
        pull-models \
        ingest process embed index \
        test test-unit test-integration test-e2e \
        lint format \
        mlflow-ui status

# ── Cores para output ──────────────────────────────────────────
BOLD  := \033[1m
CYAN  := \033[36m
GREEN := \033[32m
YELLOW:= \033[33m
RED   := \033[31m
RESET := \033[0m

help: ## Mostra esta ajuda
	@echo ""
	@echo "$(BOLD)$(CYAN)RAG Enterprise Platform$(RESET)"
	@echo "──────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ═══════════════════════════════════════════════════════════════
#  INFRAESTRUTURA
# ═══════════════════════════════════════════════════════════════

env: ## Cria .env a partir do .env.example se não existir
	@[ -f .env ] || cp .env.example .env && echo "$(GREEN)✔ .env criado$(RESET)"

up: env ## Sobe todos os serviços
	@echo "$(CYAN)▶ Subindo todos os serviços...$(RESET)"
	docker compose up -d --build
	@$(MAKE) status

down: ## Para todos os serviços
	@echo "$(YELLOW)▶ Parando serviços...$(RESET)"
	docker compose down

restart: down up ## Reinicia todos os serviços

build: ## Rebuilda imagens sem cache
	docker compose build --no-cache

logs: ## Mostra logs de todos os serviços
	docker compose logs -f

logs-%: ## Logs de um serviço específico: make logs-api
	docker compose logs -f $*

status: ## Status dos containers
	@echo ""
	@echo "$(BOLD)Status dos serviços:$(RESET)"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
	@echo ""

infra-up: env ## Sobe apenas infraestrutura (sem api/frontend)
	docker compose up -d minio postgres etcd milvus ollama mlflow minio_init

infra-down: ## Para apenas infraestrutura
	docker compose stop minio postgres etcd milvus ollama mlflow

clean: ## Remove containers, volumes e imagens do projeto
	@echo "$(RED)▶ Removendo tudo...$(RESET)"
	docker compose down -v --remove-orphans
	docker system prune -f

# ═══════════════════════════════════════════════════════════════
#  MODELOS OLLAMA
# ═══════════════════════════════════════════════════════════════

pull-models: ## Baixa modelos LLM e embedding no Ollama
	@echo "$(CYAN)▶ Baixando modelos Ollama...$(RESET)"
	docker exec rag_ollama ollama pull llama3.2
	docker exec rag_ollama ollama pull nomic-embed-text
	@echo "$(GREEN)✔ Modelos prontos$(RESET)"

list-models: ## Lista modelos disponíveis no Ollama
	docker exec rag_ollama ollama list

# ═══════════════════════════════════════════════════════════════
#  PIPELINE RAG
# ═══════════════════════════════════════════════════════════════

ingest: ## Executa ingestão de dados → Bronze
	@echo "$(CYAN)▶ Ingestão Bronze...$(RESET)"
	docker compose exec api python -m pipeline.ingestion.ingest

process: ## Processa dados Bronze → Silver → Gold
	@echo "$(CYAN)▶ Processamento Medallion...$(RESET)"
	docker compose exec api python -m pipeline.processing.bronze_to_silver
	docker compose exec api python -m pipeline.processing.silver_to_gold

embed: ## Gera embeddings e indexa no Milvus
	@echo "$(CYAN)▶ Gerando embeddings...$(RESET)"
	docker compose exec api python -m pipeline.embedding.embed_and_index

pipeline: ingest process embed ## Executa pipeline completo

# ═══════════════════════════════════════════════════════════════
#  BANCO DE DADOS
# ═══════════════════════════════════════════════════════════════

db-migrate: ## Aplica migrações do PostgreSQL
	docker compose exec api alembic upgrade head

db-rollback: ## Reverte última migração
	docker compose exec api alembic downgrade -1

db-shell: ## Abre shell do PostgreSQL
	docker compose exec postgres psql -U $${POSTGRES_USER:-raguser} $${POSTGRES_DB:-ragdb}

# ═══════════════════════════════════════════════════════════════
#  TESTES
# ═══════════════════════════════════════════════════════════════

test: ## Executa todos os testes
	docker compose exec api pytest tests/ -v --tb=short

test-unit: ## Testes unitários
	docker compose exec api pytest tests/unit/ -v

test-integration: ## Testes de integração
	docker compose exec api pytest tests/integration/ -v

test-e2e: ## Testes end-to-end
	docker compose exec api pytest tests/e2e/ -v

test-cov: ## Testes com cobertura
	docker compose exec api pytest tests/ --cov=. --cov-report=html

# ═══════════════════════════════════════════════════════════════
#  QUALIDADE DE CÓDIGO
# ═══════════════════════════════════════════════════════════════

lint: ## Roda linter (ruff)
	docker compose exec api ruff check .

format: ## Formata código (ruff + black)
	docker compose exec api ruff format .
	docker compose exec api black .

# ═══════════════════════════════════════════════════════════════
#  MLOPS
# ═══════════════════════════════════════════════════════════════

mlflow-ui: ## Abre MLflow UI no browser
	@echo "$(GREEN)MLflow: http://localhost:5000$(RESET)"
	@open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
#  ACESSO RÁPIDO AOS SERVIÇOS
# ═══════════════════════════════════════════════════════════════

open-all: ## Mostra URLs de todos os serviços
	@echo ""
	@echo "$(BOLD)$(CYAN)Serviços disponíveis:$(RESET)"
	@echo "  $(GREEN)API (FastAPI docs)$(RESET)    → http://localhost:8000/docs"
	@echo "  $(GREEN)Frontend (Gradio)$(RESET)     → http://localhost:7860"
	@echo "  $(GREEN)MLflow$(RESET)                → http://localhost:5000"
	@echo "  $(GREEN)MinIO Console$(RESET)         → http://localhost:9001"
	@echo "  $(GREEN)Milvus$(RESET)                → localhost:19530"
	@echo "  $(GREEN)PostgreSQL$(RESET)            → localhost:5432"
	@echo ""