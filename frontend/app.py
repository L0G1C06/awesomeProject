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
TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", 5))
TOP_K_MIN     = 1
TOP_K_MAX     = 10

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

/* ── Model badge ── */
.model-badge { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.45rem 0.75rem; font-size: 0.76rem; color: #6e7681; margin-top: 0.6rem; }
.model-badge code { font-family: 'DM Mono', monospace !important; color: #79c0ff !important; font-size: 0.76rem !important; }

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


# ── Query ──────────────────────────────────────────────────────────────────────
def query_rag(question: str, top_k_raw) -> tuple[str, str, object]:
    if not question.strip():
        return "⚠ Digite uma pergunta antes de consultar.", "", gr.Accordion(open=False)

    top_k, _ = validate_top_k(top_k_raw)

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_URL}/query/",
                json={"query": question, "top_k": top_k},
            )
            response.raise_for_status()
            data = response.json()

        answer  = data.get("answer", "")
        latency = data.get("latency_ms", 0)
        model   = data.get("llm_model", HF_MODEL)
        docs    = data.get("retrieved_docs", [])

        n = len(docs)
        docs_md = (
            f"**{n} documento{'s' if n != 1 else ''} recuperado{'s' if n != 1 else ''}**"
            f" &nbsp;·&nbsp; `{model}`"
            f" &nbsp;·&nbsp; {latency} ms\n\n"
        )

        for i, doc in enumerate(docs, 1):
            score    = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            title    = (
                metadata.get("title")
                or metadata.get("source")
                or metadata.get("filename")
                or f"Documento {i}"
            )

            docs_md += f"**{title}**\n\n"

            meta_parts = [f"Relevância `{score:.3f}`"]
            if metadata.get("authors"):
                meta_parts.append(f"Autores: {metadata['authors']}")
            if metadata.get("year") or metadata.get("date"):
                meta_parts.append(str(metadata.get("year") or metadata.get("date")))
            docs_md += " &nbsp;·&nbsp; ".join(meta_parts) + "\n\n"

            docs_md += f"{doc.get('content', '')}\n\n"
            if i < n:
                docs_md += "---\n\n"

        return answer, docs_md, gr.Accordion(open=True)

    except httpx.ConnectError:
        msg = "❌ Não foi possível conectar à API. Verifique se o backend está rodando."
        return msg, "", gr.Accordion(open=False)
    except httpx.HTTPStatusError as e:
        return f"❌ Erro HTTP {e.response.status_code}: {e.response.text}", "", gr.Accordion(open=False)
    except Exception as e:
        return f"❌ Erro inesperado: {str(e)}", "", gr.Accordion(open=False)


# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Enterprise", css=CSS) as demo:

    gr.HTML("""
        <div class="rag-header">
            <h1>RAG Enterprise</h1>
            <p>Consulta semântica em artigos científicos &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; Milvus &nbsp;·&nbsp; MLflow</p>
        </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=2, min_width=300):

            question_input = gr.Textbox(
                label="Pergunta",
                placeholder="Ex: How does self-attention work in transformer models?",
                lines=4,
                max_lines=10,
            )

            top_k_input = gr.Number(
                label=f"Documentos a recuperar  (mín. {TOP_K_MIN} · máx. {TOP_K_MAX})",
                value=TOP_K_DEFAULT,
                precision=0,
                elem_classes=["top-k-wrap"],
            )
            validation_msg = gr.Markdown(value="", elem_classes=["validation-error"])

            gr.HTML(f'<div class="model-badge">Modelo ativo &nbsp;·&nbsp; <code>{HF_MODEL}</code></div>')

            submit_btn = gr.Button("Consultar", variant="primary", elem_classes=["submit-btn"])

        with gr.Column(scale=3):
            answer_output = gr.Markdown(
                value="",
                label="Resposta",
                elem_classes=["answer-panel"],
            )

    with gr.Accordion("Documentos recuperados", open=False) as docs_accordion:
        docs_output = gr.Markdown(value="", elem_classes=["docs-content"])

    # ── Validação ao editar top-k ──────────────────────────────────────────────
    def on_top_k_blur(value):
        """Valida apenas ao sair do campo. Campo vazio não dispara erro."""
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
            inputs=[question_input, top_k_input],
            outputs=[answer_output, docs_output, docs_accordion],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )