import math
import re
from pathlib import Path
import torch
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from app.live_fetcher import LiveCulinaryFetcher, parse_substitution_intent

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
        self.live_fetcher = LiveCulinaryFetcher()

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.sub_collection = self.client.get_or_create_collection(
            "flavor_substitutions", metadata={"hnsw:space": "cosine"}
        )
        self.pair_collection = self.client.get_or_create_collection(
            "flavor_pairings", metadata={"hnsw:space": "cosine"}
        )

    def _upsert_substitution_cards(self, cards: list[dict]) -> None:
        if not cards:
            return
        ids, docs, metas = [], [], []
        for r in cards:
            tastes_str = ", ".join(r["tastes"])
            doc = (
                f"Replaces: {r['target']}\n"
                f"Substitute: {r['substitute']}\n"
                f"Taste Dimensions: {tastes_str}\n"
                f"Culinary Corner: {r['cuisine']}\n"
                f"Culinary & Chemical Role: {r['role']}\n"
                f"Flavor Profile: {r['flavor_profile']}\n"
                f"Key Compounds: {', '.join(r['compounds'])}\n"
                f"Best Dish Contexts: {', '.join(r['best_in'])}\n"
                f"Avoid In: {', '.join(r['avoid_in'])}"
            )
            ids.append(r["id"])
            docs.append(doc)
            metas.append(
                {
                    "target": r["target"],
                    "substitute": r["substitute"],
                    "tastes": tastes_str,
                    "cuisine": r["cuisine"],
                    "vegan": bool(r["vegan"]),
                    "source": "live_web",
                }
            )
        vecs = self.bi_encoder.encode(docs, normalize_embeddings=True).tolist()
        self.sub_collection.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)

    def search_substitutes(
        self,
        query: str,
        dish_context: str | None = None,
        taste_filter: str | None = None,
        cuisine_filter: str | None = None,
        vegan_only: bool = False,
        force_live: bool = False,
        top_k: int = 4,
    ) -> tuple[list[dict], bool]:
        target_term, parsed_dish = parse_substitution_intent(query, dish_context)
        effective_dish = dish_context or parsed_dish

        parts = [query]
        if effective_dish:
            parts.append(f"dish context: {effective_dish}")
        if taste_filter and taste_filter != "All Tastes":
            parts.append(f"taste profile: {taste_filter}")
        if cuisine_filter and cuisine_filter != "All Corners":
            parts.append(f"cuisine: {cuisine_filter}")
        search_text = " | ".join(parts)

        # Check whether target is missing from local ChromaDB or force_live is toggled
        local_results = self._query_local_subs(search_text, taste_filter, cuisine_filter, vegan_only, top_k)
        top_score = local_results[0]["relevance_score"] if local_results else 0.0
        target_matched = any(target_term.lower() in r["target"].lower() for r in local_results)

        fetched_live = False
        if force_live or (top_score < 0.68 and not target_matched):
            live_cards = self.live_fetcher.fetch_live_substitution(query, effective_dish)
            if live_cards:
                self._upsert_substitution_cards(live_cards)
                fetched_live = True
                local_results = self._query_local_subs(
                    search_text,
                    taste_filter if taste_filter != "All Tastes" else None,
                    None if cuisine_filter == "Live Web Atlas" else cuisine_filter,
                    vegan_only,
                    top_k,
                )

        return local_results, fetched_live

    def _query_local_subs(
        self,
        search_text: str,
        taste_filter: str | None,
        cuisine_filter: str | None,
        vegan_only: bool,
        top_k: int,
    ) -> list[dict]:
        total_docs = self.sub_collection.count()
        if total_docs == 0:
            return []

        q_vec = self.bi_encoder.encode([search_text], normalize_embeddings=True).tolist()
        where_filter = {"vegan": True} if vegan_only else None
        raw = self.sub_collection.query(
            query_embeddings=q_vec,
            n_results=min(30, total_docs),
            where=where_filter,
        )

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        if not docs:
            return []

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
            return []

        scores = self.cross_encoder.predict([(search_text, doc) for doc, _ in filtered])
        ranked = sorted(zip(filtered, scores), key=lambda item: float(item[1]), reverse=True)

        results = []
        for (doc, meta), score in ranked:
            prob = _sigmoid(score)
            if prob < 0.12:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "target": meta.get("target", ""),
                    "substitute": meta.get("substitute", ""),
                    "tastes": meta.get("tastes", ""),
                    "cuisine": meta.get("cuisine", ""),
                    "vegan": bool(meta.get("vegan", False)),
                    "source": meta.get("source", "local_atlas"),
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
        force_live: bool = False,
        top_k: int = 6,
    ) -> tuple[list[dict], bool]:
        normalized_inputs = [item.strip().lower() for item in ingredients if item.strip()]
        fetched_live = False

        # Auto-fetch any unknown input ingredient into ChromaDB in real time
        existing_meta = self.pair_collection.get(include=["metadatas"]).get("metadatas") or []
        known_ings = {m.get("ingredient", "").lower() for m in existing_meta}

        for ing in normalized_inputs:
            if force_live or ing not in known_ings:
                live_profile = self.live_fetcher.fetch_live_pairing(ing)
                if live_profile:
                    tastes_str = ", ".join(live_profile["tastes"])
                    doc = (
                        f"{ing.title()} brings {tastes_str} taste and pairs well with "
                        f"{', '.join(live_profile['partners'][:6])} in dishes like {', '.join(live_profile['dishes'][:3])}.\n"
                        f"Ingredient: {ing}\n"
                        f"Taste Dimensions: {tastes_str}\n"
                        f"Culinary Corner: {live_profile['cuisine']}\n"
                        f"Top Culinary Pairing Partners: {', '.join(live_profile['partners'])}\n"
                        f"Best Dish Contexts: {', '.join(live_profile['dishes'])}\n"
                        f"Live Culinary Notes: {live_profile['summary']}"
                    )
                    vec = self.bi_encoder.encode([doc], normalize_embeddings=True).tolist()
                    slug = re.sub(r"[^a-z0-9]+", "_", ing)[:40]
                    self.pair_collection.upsert(
                        ids=[f"live_ing_{slug}"],
                        documents=[doc],
                        embeddings=vec,
                        metadatas=[
                            {
                                "ingredient": ing,
                                "tastes": tastes_str,
                                "cuisine": live_profile["cuisine"],
                                "partners": ", ".join(live_profile["partners"][:8]),
                                "compounds": ", ".join(live_profile["compounds"][:4]),
                            }
                        ],
                    )
                    fetched_live = True

        total_docs = self.pair_collection.count()
        if total_docs == 0:
            return [], fetched_live

        prompt_parts = [
            f"Ingredients that pair harmoniously with {', '.join(normalized_inputs)} through shared flavor compounds and culinary co-occurrence"
        ]
        if dish_style:
            prompt_parts.append(f"for {dish_style}")
        if target_taste and target_taste != "All Tastes":
            prompt_parts.append(f"adding {target_taste} taste dimension")
        if cuisine_filter and cuisine_filter != "All Corners":
            prompt_parts.append(f"in {cuisine_filter} cooking")

        prompt = " ".join(prompt_parts)
        q_vec = self.bi_encoder.encode([prompt], normalize_embeddings=True).tolist()
        raw = self.pair_collection.query(query_embeddings=q_vec, n_results=min(35, total_docs))

        input_set = set(normalized_inputs)
        filtered = []
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0]):
            if meta.get("ingredient", "").lower() in input_set:
                continue
            if target_taste and target_taste != "All Tastes":
                if target_taste.lower() not in meta.get("tastes", "").lower():
                    continue
            if cuisine_filter and cuisine_filter != "All Corners":
                if cuisine_filter.lower() != meta.get("cuisine", "").lower():
                    continue
            filtered.append((doc, meta))

        if not filtered:
            return [], fetched_live

        scores = self.cross_encoder.predict([(prompt, doc) for doc, _ in filtered])
        ranked = sorted(zip(filtered, scores), key=lambda item: float(item[1]), reverse=True)[:top_k]

        results = [
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
        return results, fetched_live

    def find_flavor_bridges(
        self,
        pantry: list[str],
        dish_goal: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        pantry_clean = [x.strip().lower() for x in pantry if x.strip()]
        pantry_set = set(pantry_clean)
        if not pantry_clean:
            return []

        records = self.pair_collection.get(include=["metadatas", "documents"])
        metas = records.get("metadatas") or []
        docs = records.get("documents") or []
        if not metas:
            return []

        candidates = []
        for doc, meta in zip(docs, metas):
            ing = meta.get("ingredient", "").strip().lower()
            if not ing or ing in pantry_set:
                continue
            partners = {x.strip().lower() for x in meta.get("partners", "").split(",") if x.strip()}
            bridged = sorted(pantry_set.intersection(partners))
            candidates.append((ing, doc, meta, bridged))

        prompt = (
            f"Single bridge ingredient that unites {', '.join(pantry_clean)} "
            f"into a cohesive dish through shared aroma compounds and culinary pairing"
        )
        if dish_goal:
            prompt += f" ({dish_goal})"

        scores = self.cross_encoder.predict([(prompt, doc) for _, doc, _, _ in candidates])

        scored = []
        for (ing, doc, meta, bridged), logit in zip(candidates, scores):
            prob = _sigmoid(logit)
            # Combine neural relevance with multi-ingredient graph connectivity
            bridge_bonus = 0.15 * (len(bridged) / max(1, len(pantry_clean)))
            combined = round(min(0.9999, prob + bridge_bonus), 4)
            scored.append((combined, prob, ing, doc, meta, bridged))

        scored.sort(key=lambda x: (len(x[5]), x[0]), reverse=True)

        return [
            {
                "rank": idx,
                "ingredient": ing,
                "bridged_items": bridged,
                "bridge_coverage": f"{len(bridged)}/{len(pantry_clean)}",
                "tastes": meta.get("tastes", ""),
                "cuisine": meta.get("cuisine", ""),
                "compounds": meta.get("compounds", ""),
                "relevance_score": combined,
                "details": doc,
            }
            for idx, (combined, _, ing, doc, meta, bridged) in enumerate(scored[:top_k], start=1)
        ]
