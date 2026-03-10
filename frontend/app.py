"""
Frontend — Gradio
Interface de consulta RAG com visualização de documentos recuperados.
"""
import os
import httpx
import gradio as gr

API_URL = os.getenv("API_URL", "http://api:8000")


def query_rag(question: str, top_k: int, model: str) -> tuple[str, str]:
    """Envia query para a API RAG e retorna resposta + contexto."""
    if not question.strip():
        return "Por favor, digite uma pergunta.", ""

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_URL}/query/",
                json={"query": question, "top_k": int(top_k), "llm_model": model},
            )
            response.raise_for_status()
            data = response.json()

        answer = data["answer"]
        latency = data.get("latency_ms", 0)

        # Formata os documentos recuperados
        docs_text = f"**{len(data['retrieved_docs'])} documentos recuperados** | Latência: {latency}ms\n\n"
        for i, doc in enumerate(data["retrieved_docs"], 1):
            score = doc.get("score", 0)
            docs_text += f"### 📄 Documento {i} (score: {score:.3f})\n"
            docs_text += f"{doc['content']}\n\n"
            if doc.get("metadata"):
                docs_text += f"*Metadados: {doc['metadata']}*\n\n"
            docs_text += "---\n"

        return answer, docs_text

    except httpx.ConnectError:
        return "❌ Erro: Não foi possível conectar à API. Verifique se está rodando.", ""
    except Exception as e:
        return f"❌ Erro: {str(e)}", ""


def submit_feedback(run_id: str, feedback: int):
    """Envia feedback para a API."""
    if not run_id:
        return "Faça uma consulta primeiro."
    try:
        with httpx.Client() as client:
            client.post(f"{API_URL}/query/feedback", params={"run_id": run_id, "feedback": feedback})
        labels = {1: "👍 Positivo", -1: "👎 Negativo", 0: "😐 Neutro"}
        return f"Feedback registrado: {labels.get(feedback, str(feedback))}"
    except Exception as e:
        return f"Erro ao registrar feedback: {e}"


# ── Layout Gradio ───────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="RAG Enterprise",
) as demo:
    gr.Markdown(
        """
        # 🔍 RAG Enterprise Platform
        ### Consulte documentos com IA local (Ollama + Milvus)
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Sua pergunta",
                placeholder="Ex: Quais são os padrões mais comuns encontrados nos dados?",
                lines=3,
            )
            with gr.Row():
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Documentos a recuperar (top-k)")
                model_select = gr.Dropdown(
                    choices=["llama3.2", "llama3.1", "mistral", "phi3"],
                    value="llama3.2",
                    label="Modelo LLM",
                )
            submit_btn = gr.Button("🔍 Consultar", variant="primary", size="lg")

        with gr.Column(scale=3):
            answer_output = gr.Markdown(label="Resposta gerada")

    with gr.Accordion("📚 Documentos Recuperados", open=False):
        docs_output = gr.Markdown()

    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("**Avalie esta resposta:**")
        feedback_positive = gr.Button("👍")
        feedback_neutral  = gr.Button("😐")
        feedback_negative = gr.Button("👎")
        feedback_status   = gr.Textbox(label="", interactive=False, max_lines=1)

    run_id_state = gr.State("")

    def handle_query(q, k, m):
        answer, docs = query_rag(q, k, m)
        return answer, docs

    submit_btn.click(
        fn=handle_query,
        inputs=[question_input, top_k_slider, model_select],
        outputs=[answer_output, docs_output],
    )

    feedback_positive.click(fn=lambda rid: submit_feedback(rid, 1),  inputs=[run_id_state], outputs=[feedback_status])
    feedback_neutral.click(fn=lambda rid: submit_feedback(rid, 0),   inputs=[run_id_state], outputs=[feedback_status])
    feedback_negative.click(fn=lambda rid: submit_feedback(rid, -1), inputs=[run_id_state], outputs=[feedback_status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )