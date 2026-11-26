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
        taste_filter: str | None = None,
        cuisine_filter: str | None = None,
        vegan_only: bool = False,
        top_k: int = 4,
    ) -> list[dict]:
        total_docs = self.sub_collection.count()
        if total_docs == 0:
            return []

        parts = [query]
        if dish_context:
            parts.append(f"dish context: {dish_context}")
        if taste_filter and taste_filter != "All Tastes":
            parts.append(f"taste profile: {taste_filter}")
        if cuisine_filter and cuisine_filter != "All Corners":
            parts.append(f"cuisine: {cuisine_filter}")
        search_text = " | ".join(parts)

        q_vec = self.bi_encoder.encode([search_text], normalize_embeddings=True).tolist()
        where_filter = {"vegan": True} if vegan_only else None

        raw = self.sub_collection.query(
            query_embeddings=q_vec,
            n_results=min(25, total_docs),
            where=where_filter,
        )

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        if not docs:
            return []

        # Optional post-filtering by taste or cuisine corner
        filtered = []
        for doc, meta in zip(docs, metas):
            if taste_filter and taste_filter != "All Tastes":
                if taste_filter.lower() not in meta.get("tastes", "").lower():
                    continue
            if cuisine_filter and cuisine_filter != "All Corners":
                if cuisine_filter.lower() != meta.get("cuisine", "").lower():
                    continue
            filtered.append((doc, meta))

        if not filtered:
            filtered = list(zip(docs, metas))

        scores = self.cross_encoder.predict([(search_text, doc) for doc, _ in filtered])
        ranked = sorted(
            zip(filtered, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []
        for (doc, meta), score in ranked:
            prob = _sigmoid(score)
            if prob < 0.08 and results:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "target": meta.get("target", ""),
                    "substitute": meta.get("substitute", ""),
                    "tastes": meta.get("tastes", ""),
                    "cuisine": meta.get("cuisine", ""),
                    "vegan": bool(meta.get("vegan", False)),
                    "relevance_score": prob,
                    "details": doc,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def suggest_pairings(
        self,
        ingredients: list[str],
        dish_style: str | None = None,
        target_taste: str | None = None,
        cuisine_filter: str | None = None,
        top_k: int = 6,
    ) -> list[dict]:
        total_docs = self.pair_collection.count()
        if total_docs == 0:
            return []

        normalized_inputs = {item.strip().lower() for item in ingredients}
        prompt_parts = [
            f"Ingredients that pair harmoniously with {', '.join(ingredients)} through shared flavor compounds and culinary co-occurrence"
        ]
        if dish_style:
            prompt_parts.append(f"for {dish_style}")
        if target_taste and target_taste != "All Tastes":
            prompt_parts.append(f"adding {target_taste} taste dimension")
        if cuisine_filter and cuisine_filter != "All Corners":
            prompt_parts.append(f"in {cuisine_filter} cooking")

        prompt = " ".join(prompt_parts)
        q_vec = self.bi_encoder.encode([prompt], normalize_embeddings=True).tolist()

        raw = self.pair_collection.query(query_embeddings=q_vec, n_results=min(30, total_docs))
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]

        filtered = []
        for doc, meta in zip(docs, metas):
            if meta.get("ingredient", "").lower() in normalized_inputs:
                continue
            if target_taste and target_taste != "All Tastes":
                if target_taste.lower() not in meta.get("tastes", "").lower():
                    continue
            if cuisine_filter and cuisine_filter != "All Corners":
                if cuisine_filter.lower() != meta.get("cuisine", "").lower():
                    continue
            filtered.append((doc, meta))

        if not filtered:
            filtered = [
                (doc, meta)
                for doc, meta in zip(docs, metas)
                if meta.get("ingredient", "").lower() not in normalized_inputs
            ]

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
                "tastes": meta.get("tastes", ""),
                "cuisine": meta.get("cuisine", ""),
                "relevance_score": _sigmoid(score),
                "common_partners": meta.get("partners", ""),
                "compounds": meta.get("compounds", ""),
                "details": doc,
            }
            for idx, ((doc, meta), score) in enumerate(ranked, start=1)
        ]
