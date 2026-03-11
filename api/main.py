"""
RAG Enterprise API — Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routers import query

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

# ── Routers ────────────────────────────────────────────────────
app.include_router(query.router, prefix="/query", tags=["RAG Query"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
