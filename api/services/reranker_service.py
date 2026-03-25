from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self):
        self.model = CrossEncoder("models/reranker")

    def rerank(self, query: str, docs: list, top_k: int = 3):
        pairs = [(query, doc.content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]