"""
Frontend — Gradio
Interface de consulta RAG com visualização de documentos recuperados.
"""
import os
import httpx
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_URL   = os.getenv("API_URL", "http://localhost:8001")
HF_MODEL  = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", 5))

current_run_id = {"value": None}


def query_rag(question: str, top_k: int) -> tuple[str, str, str]:
    """Envia query para a API RAG e retorna resposta + contexto + run_id."""
    if not question.strip():
        return "Por favor, digite uma pergunta.", "", ""

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_URL}/query/",
                json={"query": question, "top_k": int(top_k)},
            )
            response.raise_for_status()
            data = response.json()

        run_id  = data.get("run_id", "")
        answer  = data.get("answer", "")
        latency = data.get("latency_ms", 0)
        model   = data.get("llm_model", HF_MODEL)

        current_run_id["value"] = run_id

        # Formata os documentos recuperados
        docs_text = (
            f"**{len(data['retrieved_docs'])} documentos recuperados** "
            f"| Modelo: `{model}` "
            f"| Latência: {latency}ms\n\n"
        )
        for i, doc in enumerate(data["retrieved_docs"], 1):
            score = doc.get("score", 0)
            docs_text += f"### 📄 Documento {i} (score: {score:.3f})\n"
            docs_text += f"{doc['content']}\n\n"
            if doc.get("metadata"):
                docs_text += f"*Metadados: {doc['metadata']}*\n\n"
            docs_text += "---\n"

        return answer, docs_text, run_id

    except httpx.ConnectError:
        return "❌ Erro: Não foi possível conectar à API. Verifique se está rodando.", "", ""
    except httpx.HTTPStatusError as e:
        return f"❌ Erro HTTP {e.response.status_code}: {e.response.text}", "", ""
    except Exception as e:
        return f"❌ Erro inesperado: {str(e)}", "", ""


def submit_feedback(run_id: str, feedback: int) -> str:
    """Envia feedback para a API."""
    rid = run_id or current_run_id["value"]
    if not rid:
        return "⚠️ Faça uma consulta primeiro."
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{API_URL}/query/feedback",
                params={"run_id": rid, "feedback": feedback},
            )
        labels = {1: "👍 Positivo", -1: "👎 Negativo", 0: "😐 Neutro"}
        return f"Feedback registrado: {labels.get(feedback, str(feedback))}"
    except Exception as e:
        return f"❌ Erro ao registrar feedback: {e}"


# ── Layout Gradio ──────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Enterprise") as demo:
    gr.Markdown(
        """
        # 🔍 RAG Enterprise Platform
        ### Consulte artigos científicos com IA · HuggingFace + Milvus + MLflow
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Sua pergunta",
                placeholder="Ex: How does self-attention work in transformer models?",
                lines=3,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=20,
                value=TOP_K_DEFAULT,
                step=1,
                label=f"Documentos a recuperar (top-k) — padrão: {TOP_K_DEFAULT}",
            )
            gr.Markdown(f"🤖 Modelo ativo: `{HF_MODEL}`")
            submit_btn = gr.Button("🔍 Consultar", variant="primary", size="lg")

        with gr.Column(scale=3):
            answer_output = gr.Markdown(label="Resposta gerada")

    with gr.Accordion("📚 Documentos Recuperados", open=False):
        docs_output = gr.Markdown()

    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("**Avalie esta resposta:**")
        feedback_positive = gr.Button("👍 Útil")
        feedback_neutral  = gr.Button("😐 Neutro")
        feedback_negative = gr.Button("👎 Não útil")
        feedback_status   = gr.Textbox(label="", interactive=False, max_lines=1)

    run_id_state = gr.State("")

    submit_btn.click(
        fn=query_rag,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, docs_output, run_id_state],
    )

    feedback_positive.click(
        fn=lambda rid: submit_feedback(rid, 1),
        inputs=[run_id_state],
        outputs=[feedback_status],
    )
    feedback_neutral.click(
        fn=lambda rid: submit_feedback(rid, 0),
        inputs=[run_id_state],
        outputs=[feedback_status],
    )
    feedback_negative.click(
        fn=lambda rid: submit_feedback(rid, -1),
        inputs=[run_id_state],
        outputs=[feedback_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )