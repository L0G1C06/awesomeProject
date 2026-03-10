"""
RAG Enterprise API — Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routers import health, query, ingest, documents, experiments
from api.middleware.logging import LoggingMiddleware

app = FastAPI(
    title="RAG Enterprise API",
    description="Plataforma RAG com Governança Medallion",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# ── Routers ────────────────────────────────────────────────────
app.include_router(health.router,       tags=["Health"])
app.include_router(query.router,        prefix="/query",       tags=["RAG Query"])
app.include_router(ingest.router,       prefix="/ingest",      tags=["Ingestão"])
app.include_router(documents.router,    prefix="/documents",   tags=["Documentos"])
app.include_router(experiments.router,  prefix="/experiments", tags=["MLflow"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")