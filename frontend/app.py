"""
Frontend — Gradio
Interface de consulta RAG com visualização de documentos recuperados.
"""
import os
import httpx
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_URL       = os.getenv("API_URL", "http://localhost:8001")
HF_MODEL      = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "huggingface").lower()
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
ACTIVE_MODEL  = OPENAI_MODEL if LLM_PROVIDER == "openai" else HF_MODEL
GRADIO_PORT   = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", 5))
TOP_K_MIN     = 1
TOP_K_MAX     = 20

# ── Modelos disponíveis ────────────────────────────────────────────────────────
# Modelos Ollama (locais, rodando em container)
OLLAMA_MODELS = [
    "tinyllama",
]

# Modelos HuggingFace (via API remota)
HUGGINGFACE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

# Modelos OpenAI (via API)
OPENAI_MODELS = [
    "gpt-5.4-mini"
]

# Combina os modelos com rótulos visuais para distinguir
AVAILABLE_MODELS = (
    [f"🔵 {m} [ollama]" for m in OLLAMA_MODELS] +
    [f"🟠 {m} [huggingface]" for m in HUGGINGFACE_MODELS] +
    [f"🔴 {m} [openai]" for m in OPENAI_MODELS]
)

# Garante que o modelo padrão do .env esteja na lista
DEFAULT_DISPLAY = None
if OPENAI_MODEL and LLM_PROVIDER == "openai":
    for model in AVAILABLE_MODELS:
        if OPENAI_MODEL in model:
            DEFAULT_DISPLAY = model
            break
    else:
        if OPENAI_MODEL.startswith("gpt"):
            DEFAULT_DISPLAY = f"🔴 {OPENAI_MODEL} [openai]"
elif HF_MODEL:
    for model in AVAILABLE_MODELS:
        if HF_MODEL in model:
            DEFAULT_DISPLAY = model
            break
    else:
        # Se não encontrando, adiciona com a marcação correta
        if "/" in HF_MODEL:
            # É um modelo HuggingFace
            DEFAULT_DISPLAY = f"🟠 {HF_MODEL} [huggingface]"
            AVAILABLE_MODELS.insert(0, DEFAULT_DISPLAY)
        else:
            # É um modelo Ollama
            DEFAULT_DISPLAY = f"🔵 {HF_MODEL} [ollama]"
            AVAILABLE_MODELS.insert(0, DEFAULT_DISPLAY)

if not DEFAULT_DISPLAY:
    DEFAULT_DISPLAY = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else HF_MODEL

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { font-family: 'DM Sans', sans-serif !important; background: #0d1117 !important; color: #c9d1d9 !important; }

/* ── Header ── */
.rag-header { padding: 2rem 0 1.5rem; border-bottom: 1px solid #21262d; margin-bottom: 1.75rem; }
.rag-header h1 { margin: 0 !important; font-size: 1.45rem !important; font-weight: 600 !important; color: #f0f6fc !important; letter-spacing: -0.025em; }
.rag-header p  { margin: 0.3rem 0 0 !important; font-size: 0.82rem !important; color: #6e7681 !important; }

/* ── Labels ── */
label > span { font-size: 0.75rem !important; font-weight: 500 !important; color: #8b949e !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }

/* ── Textbox ── */
.gr-textbox textarea, .gr-textbox input {
    background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important;
    color: #c9d1d9 !important; font-size: 0.9rem !important; padding: 0.7rem 0.9rem !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #388bfd !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,.15) !important;
}

/* ── Number input ── */
.top-k-wrap input[type=number] {
    background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important;
    color: #c9d1d9 !important; font-size: 0.9rem !important; padding: 0.7rem 0.9rem !important; width: 100% !important;
    transition: border-color 0.15s;
}
.top-k-wrap input[type=number]:focus { border-color: #388bfd !important; outline: none !important; }
.validation-error { font-size: 0.76rem !important; color: #f85149 !important; min-height: 1rem; margin-top: 0.2rem; }

/* ── Model selector dropdown ── */
.model-selector-wrap .wrap { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; transition: border-color 0.15s; }
.model-selector-wrap .wrap:focus-within { border-color: #388bfd !important; box-shadow: 0 0 0 3px rgba(56,139,253,.15) !important; }
.model-selector-wrap input { background: transparent !important; border: none !important; color: #c9d1d9 !important; font-size: 0.86rem !important; font-family: 'DM Mono', monospace !important; }
.model-selector-wrap svg { color: #6e7681 !important; }
.model-selector-wrap ul { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; margin-top: 4px !important; z-index: 100 !important; }
.model-selector-wrap ul li { color: #c9d1d9 !important; font-size: 0.84rem !important; font-family: 'DM Mono', monospace !important; padding: 0.5rem 0.9rem !important; }
.model-selector-wrap ul li:hover { background: #21262d !important; color: #f0f6fc !important; }
.model-selector-wrap ul li.selected { background: #1f3a5f !important; color: #79c0ff !important; }

/* ── Submit button ── */
.submit-btn > button {
    background: #238636 !important; border: 1px solid #2ea043 !important; border-radius: 8px !important;
    color: #fff !important; font-size: 0.88rem !important; font-weight: 500 !important;
    padding: 0.7rem 1.25rem !important; width: 100% !important; cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
}
.submit-btn > button:hover   { background: #2ea043 !important; }
.submit-btn > button:active  { transform: scale(0.985) !important; }
.submit-btn > button:disabled { background: #161b22 !important; color: #484f58 !important; border-color: #21262d !important; cursor: not-allowed !important; }

/* ── Run meta bar ── */
.run-meta { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 0.55rem 1rem;
            font-size: 0.76rem; color: #6e7681; margin-bottom: 1rem; display: flex; gap: 1.25rem; flex-wrap: wrap; }
.run-meta code { font-family: 'DM Mono', monospace !important; color: #79c0ff !important; font-size: 0.74rem !important; }

/* ── Answer panel ── */
.answer-panel { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important; padding: 1.25rem 1.5rem !important; min-height: 220px; }
.answer-panel p, .answer-panel li { font-size: 0.91rem !important; line-height: 1.72 !important; color: #c9d1d9 !important; }
.answer-panel strong { color: #f0f6fc !important; }
.answer-panel code   { font-family: 'DM Mono', monospace !important; background: #0d1117; padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.83em !important; color: #79c0ff !important; }

/* ── Accordion ── */
.gr-accordion { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important; margin-top: 1.25rem !important; overflow: hidden; }
.gr-accordion .label-wrap { padding: 0.85rem 1.25rem !important; }
.gr-accordion .label-wrap span { font-size: 0.82rem !important; font-weight: 500 !important; color: #8b949e !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }

/* ── Docs content ── */
.docs-content p, .docs-content li { font-size: 0.87rem !important; line-height: 1.68 !important; color: #8b949e !important; }
.docs-content strong { color: #c9d1d9 !important; font-weight: 600 !important; font-size: 0.89rem !important; }
.docs-content em  { color: #6e7681 !important; }
.docs-content code { font-family: 'DM Mono', monospace !important; background: #0d1117 !important; padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.82em !important; color: #79c0ff !important; }
.docs-content hr  { border: none !important; border-top: 1px solid #21262d !important; margin: 1.1rem 0 !important; }

/* ── Tag pills (categories) ── */
.docs-content .pill {
    display: inline-block; background: #0d1117; border: 1px solid #30363d;
    border-radius: 20px; padding: 0.1em 0.55em; font-size: 0.74rem !important;
    color: #8b949e !important; margin: 0 0.2rem 0.2rem 0; line-height: 1.5;
}
"""

# ── Validação ──────────────────────────────────────────────────────────────────
def validate_top_k(value) -> tuple[int, str]:
    try:
        v = int(value)
    except (ValueError, TypeError):
        return TOP_K_DEFAULT, f"Valor inválido — usando padrão ({TOP_K_DEFAULT})"
    if v < TOP_K_MIN:
        return TOP_K_MIN, f"Mínimo permitido é {TOP_K_MIN}"
    if v > TOP_K_MAX:
        return TOP_K_MAX, f"Máximo permitido é {TOP_K_MAX}"
    return v, ""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _extract_model_name(display_name: str) -> str:
    """
    Extrai o nome real do modelo e provider a partir do nome formatado.
    Ex: "🔵 llama2 [ollama]" → ("llama2", "ollama")
    Ex: "🟠 meta-llama/Meta-Llama-3-8B-Instruct [huggingface]" → ("meta-llama/Meta-Llama-3-8B-Instruct", "huggingface")
    Ex: "🔴 gpt-4 [openai]" → ("gpt-4", "openai")
    """
    if not display_name:
        return None, None
    
    # Remove o emoji no início (3-4 bytes dependendo do emoji)
    name = display_name
    for emoji in ["🔵", "🟠", "🔴"]:
        if name.startswith(emoji + " "):
            name = name[2:].lstrip()
            break
    
    # Extrai o provider entre colchetes [provider]
    provider = None
    if "[" in name and "]" in name:
        model_part, provider_part = name.rsplit("[", 1)
        provider = provider_part.rstrip("]").strip()
        name = model_part.strip()
    
    return name, provider


def _pill(text: str) -> str:
    return f'<span class="pill">{text}</span>'


def _format_doc(i: int, doc: dict, total: int) -> str:
    score            = doc.get("score", 0)
    title            = doc.get("title") or f"Documento {i}"
    authors          = doc.get("authors")
    published        = doc.get("published")
    updated          = doc.get("updated")
    url              = doc.get("url")
    arxiv_id         = doc.get("arxiv_id")
    categories       = doc.get("categories")
    primary_category = doc.get("primary_category")
    content          = doc.get("content", "")

    lines: list[str] = []

    if url:
        lines.append(f"**[{title}]({url})**")
    else:
        lines.append(f"**{title}**")

    meta_parts = [f"Relevância `{score:.3f}`"]
    if primary_category:
        meta_parts.append(f"Categoria primária `{primary_category}`")
    if published:
        meta_parts.append(f"Publicado: {published[:10]}")
    if updated and updated != published:
        meta_parts.append(f"Atualizado: {updated[:10]}")
    lines.append(" &nbsp;·&nbsp; ".join(meta_parts))
    lines.append("")

    if authors:
        lines.append(f"*{authors}*")
        lines.append("")

    if categories:
        cats = [c.strip() for c in categories.split() if c.strip()]
        if cats:
            pills = " ".join(_pill(c) for c in cats)
            lines.append(pills)
            lines.append("")

    ids: list[str] = []
    if arxiv_id:
        ids.append(f"arXiv: `{arxiv_id}`")
    if ids:
        lines.append(" &nbsp;·&nbsp; ".join(ids))
        lines.append("")

    lines.append(content)

    if i < total:
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── Query ──────────────────────────────────────────────────────────────────────
def query_rag(question: str, top_k_raw, selected_model: str) -> tuple[str, str, str, object]:
    """Retorna: (answer_md, run_meta_html, docs_md, accordion_state)."""
    empty_meta = ""

    if not question.strip():
        return (
            "⚠ Digite uma pergunta antes de consultar.",
            empty_meta,
            "",
            gr.Accordion(open=False),
        )

    top_k, _ = validate_top_k(top_k_raw)

    # Extrai o nome real do modelo e provider a partir da exibição formatada
    model_to_use, provider_to_use = _extract_model_name(selected_model) if selected_model else (HF_MODEL, LLM_PROVIDER)
    
    if not model_to_use:
        model_to_use = HF_MODEL
    if not provider_to_use:
        provider_to_use = LLM_PROVIDER

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_URL}/query/",
                json={
                    "query": question,
                    "top_k": top_k,
                    "llm_model": model_to_use,
                    "llm_provider": provider_to_use,
                },
            )
            response.raise_for_status()
            data = response.json()

        # ── Campos de QueryResponse ──
        answer       = data.get("answer", "")
        latency      = data.get("latency_ms", 0)
        provider     = data.get("llm_provider", provider_to_use)
        model        = data.get("llm_model", model_to_use)
        docs         = data.get("retrieved_docs", [])
        run_id       = data.get("run_id", "")
        mlflow_run   = data.get("mlflow_run_id")

        n = len(docs)

        meta_parts = [
            f"<span><strong>Provider</strong> &nbsp;<code>{provider}</code></span>",
            f"<span><strong>Modelo</strong> &nbsp;<code>{model}</code></span>",
            f"<span><strong>Latência</strong> &nbsp;<code>{latency} ms</code></span>",
            f"<span><strong>Docs recuperados</strong> &nbsp;<code>{n}</code></span>",
        ]
        if run_id:
            meta_parts.append(f"<span><strong>Run ID</strong> &nbsp;<code>{run_id}</code></span>")
        if mlflow_run:
            meta_parts.append(f"<span><strong>MLflow</strong> &nbsp;<code>{mlflow_run}</code></span>")

        run_meta_html = '<div class="run-meta">' + "".join(meta_parts) + "</div>"

        docs_md = "\n".join(_format_doc(i, doc, n) for i, doc in enumerate(docs, 1))

        return answer, run_meta_html, docs_md, gr.Accordion(open=True)

    except httpx.ConnectError:
        msg = "❌ Não foi possível conectar à API. Verifique se o backend está rodando."
        return msg, empty_meta, "", gr.Accordion(open=False)
    except httpx.HTTPStatusError as e:
        return (
            f"❌ Erro HTTP {e.response.status_code}: {e.response.text}",
            empty_meta,
            "",
            gr.Accordion(open=False),
        )
    except Exception as e:
        return f"❌ Erro inesperado: {str(e)}", empty_meta, "", gr.Accordion(open=False)


# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Buscador de Documentos Científicos") as demo:

    gr.HTML("""
        <div class="rag-header">
            <h1>Buscador de Documentos Científicos</h1>
            <p>Consulta semântica em artigos científicos</p>
        </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=2, min_width=300):

            question_input = gr.Textbox(
                label="Pergunta",
                lines=4,
                max_lines=10,
            )

            # ── Seletor de modelo ──────────────────────────────────────────────
            model_dropdown = gr.Dropdown(
                label="Modelo de linguagem",
                info="🔵 Modelos locais (Ollama) | 🟠 Modelos remostos (HuggingFace) | 🔴 Modelos OpenAI",
                choices=AVAILABLE_MODELS,
                value=DEFAULT_DISPLAY,
                allow_custom_value=True,   # permite digitar um HF model ID customizado
                elem_classes=["model-selector-wrap"],
            )

            top_k_input = gr.Number(
                label=f"Documentos a recuperar  (mín. {TOP_K_MIN} · máx. {TOP_K_MAX})",
                value=TOP_K_DEFAULT,
                precision=0,
                elem_classes=["top-k-wrap"],
            )
            validation_msg = gr.Markdown(value="", elem_classes=["validation-error"])

            submit_btn = gr.Button("Consultar", variant="primary", elem_classes=["submit-btn"])

        with gr.Column(scale=3):
            run_meta_output = gr.HTML(value="")

            answer_output = gr.Markdown(
                value="",
                label="Resposta",
                elem_classes=["answer-panel"],
            )

    with gr.Accordion("Documentos recuperados", open=False) as docs_accordion:
        docs_output = gr.Markdown(value="", elem_classes=["docs-content"])

    # ── Validação ao editar top-k ──────────────────────────────────────────────
    def on_top_k_blur(value):
        if value is None or str(value).strip() == "":
            return TOP_K_DEFAULT, ""
        corrected, msg = validate_top_k(value)
        return corrected, msg

    top_k_input.blur(
        fn=on_top_k_blur,
        inputs=[top_k_input],
        outputs=[top_k_input, validation_msg],
    )

    # ── Submit (botão e Enter) ─────────────────────────────────────────────────
    for trigger in (submit_btn.click, question_input.submit):
        trigger(
            fn=query_rag,
            inputs=[question_input, top_k_input, model_dropdown],  # model_dropdown adicionado
            outputs=[answer_output, run_meta_output, docs_output, docs_accordion],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        css=CSS,
    )
