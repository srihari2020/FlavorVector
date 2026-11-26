import math
from pathlib import Path
import torch
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

BI_ENCODER_NAME = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(logit: float) -> float:
    return round(1.0 / (1.0 + math.exp(-float(logit))), 4)


class FlavorEngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device=self.device)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=self.device)

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.sub_collection = self.client.get_or_create_collection("flavor_substitutions")
        self.pair_collection = self.client.get_or_create_collection("flavor_pairings")

    def search_substitutes(
        self,
        query: str,
        dish_context: str | None = None,
        vegan_only: bool = False,
        top_k: int = 3,
    ) -> list[dict]:
        total_docs = self.sub_collection.count()
        if total_docs == 0:
            return []

        search_text = f"{query} (dish context: {dish_context})" if dish_context else query
        q_vec = self.bi_encoder.encode([search_text], normalize_embeddings=True).tolist()

        n_candidates = min(max(top_k * 3, 10), total_docs)
        where_filter = {"vegan": True} if vegan_only else None

        raw = self.sub_collection.query(
            query_embeddings=q_vec,
            n_results=n_candidates,
            where=where_filter,
        )

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        if not docs:
            return []

        scores = self.cross_encoder.predict([(search_text, doc) for doc in docs])
        ranked = sorted(
            zip(docs, metas, scores),
            key=lambda item: float(item[2]),
            reverse=True,
        )[:top_k]

        return [
            {
                "rank": idx,
                "target": meta.get("target", ""),
                "substitute": meta.get("substitute", ""),
                "vegan": bool(meta.get("vegan", False)),
                "relevance_score": _sigmoid(score),
                "details": doc,
            }
            for idx, (doc, meta, score) in enumerate(ranked, start=1)
        ]

    def suggest_pairings(
        self,
        ingredients: list[str],
        dish_style: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        total_docs = self.pair_collection.count()
        if total_docs == 0:
            return []

        normalized_inputs = {item.strip().lower() for item in ingredients}
        prompt = (
            f"Ingredients that pair harmoniously with {', '.join(ingredients)} "
            f"through shared flavor compounds and culinary co-occurrence"
        )
        if dish_style:
            prompt += f" in {dish_style}"

        q_vec = self.bi_encoder.encode([prompt], normalize_embeddings=True).tolist()
        n_candidates = min(max(top_k * 4, 20), total_docs)

        raw = self.pair_collection.query(query_embeddings=q_vec, n_results=n_candidates)
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]

        filtered = [
            (doc, meta)
            for doc, meta in zip(docs, metas)
            if meta.get("ingredient", "").lower() not in normalized_inputs
        ]
        if not filtered:
            return []

        scores = self.cross_encoder.predict([(prompt, doc) for doc, _ in filtered])
        ranked = sorted(
            zip(filtered, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        return [
            {
                "rank": idx,
                "ingredient": meta.get("ingredient", ""),
                "relevance_score": _sigmoid(score),
                "common_partners": meta.get("partners", ""),
                "compounds": meta.get("compounds", ""),
                "details": doc,
            }
            for idx, ((doc, meta), score) in enumerate(ranked, start=1)
        ]
