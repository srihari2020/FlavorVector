import json
import re
import requests

HEADERS = {"User-Agent": "FlavorVector/1.0 (Food-Science Retrieval Engine)"}

# Whole meals/breads that must never be suggested as an ingredient/condiment substitute
DISALLOWED_SUBSTITUTES = {
    "dosa", "idli", "samosa", "roti", "naan", "chapati", "paratha",
    "biryani", "pizza", "burger", "pasta", "bread", "rice", "upma", "poha",
}


def parse_substitution_intent(query: str, dish_context: str | None = None) -> tuple[str, str]:
    q = query.strip().rstrip("?.!")
    target = q
    dish = (dish_context or "").strip()

    patterns = [
        r"(?:instead of|replace|substitute for|alternative to|swap for)\s+(.+?)(?:\s+(?:in|for|with)\s+(.+))?$",
        r"what can i use (?:instead of|for)\s+(.+?)(?:\s+(?:in|for)\s+(.+))?$",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            target = m.group(1).strip()
            if not dish and m.group(2):
                dish = m.group(2).strip()
            break

    target = re.sub(r"^(?:a|an|the|some)\s+", "", target, flags=re.IGNORECASE)
    dish = re.sub(r"^(?:a|an|the|some)\s+", "", dish, flags=re.IGNORECASE)
    return target, dish


class LiveCulinaryFetcher:
    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout

    def _wiki_extract_best(self, query: str) -> tuple[str, str]:
        """Fetch the lead summary of the single closest Wikipedia article for the target ingredient."""
        url = "https://en.wikipedia.org/w/api.php"
        try:
            search_resp = requests.get(
                url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 2,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=self.timeout,
            )
            hits = search_resp.json().get("query", {}).get("search", [])
            if not hits:
                return "", ""

            title = hits[0].get("title", "")
            ext_resp = requests.get(
                url,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=self.timeout,
            )
            pages = ext_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = (page.get("extract") or "").strip()
                if text:
                    return title, text[:600]
        except Exception:
            pass
        return "", ""

    def _mealdb_cooccurrences(self, keyword: str) -> tuple[list[str], list[str]]:
        """Return (co_occurring_ingredients, dish_names) from TheMealDB."""
        co_ings: list[str] = []
        dish_names: list[str] = []
        for endpoint in (
            f"https://www.themealdb.com/api/json/v1/1/filter.php?i={requests.utils.quote(keyword)}",
            f"https://www.themealdb.com/api/json/v1/1/search.php?s={requests.utils.quote(keyword)}",
        ):
            try:
                resp = requests.get(endpoint, headers=HEADERS, timeout=self.timeout)
                meals = resp.json().get("meals") or []
                for m in meals[:5]:
                    if m.get("strMeal"):
                        dish_names.append(m["strMeal"].lower())
                    for i in range(1, 21):
                        val = (m.get(f"strIngredient{i}") or "").strip().lower()
                        if val and val != keyword.lower() and val not in co_ings:
                            co_ings.append(val)
            except Exception:
                continue
        return co_ings[:10], dish_names[:5]

    def _openfoodfacts_profile(self, term: str) -> dict:
        try:
            resp = requests.get(
                "https://world.openfoodfacts.org/cgi/search.pl",
                params={"search_terms": term, "search_simple": 1, "action": "process", "json": 1, "page_size": 3},
                headers=HEADERS,
                timeout=self.timeout,
            )
            products = resp.json().get("products") or []
            if not products:
                return {}
            p = products[0]
            labels = [lbl.lower() for lbl in (p.get("labels_tags") or [])]
            is_vegan = any("vegan" in lbl for lbl in labels) or "en:non-vegan" not in labels
            return {
                "ingredients_text": (p.get("ingredients_text") or "")[:250],
                "vegan": is_vegan,
            }
        except Exception:
            return {}

    def _try_ollama_structure(self, prompt: str) -> dict | None:
        try:
            tags = requests.get("http://localhost:11434/api/tags", timeout=1.2)
            if tags.status_code != 200:
                return None
            models = [m["name"] for m in tags.json().get("models", [])]
            if not models:
                return None
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": models[0], "prompt": prompt, "format": "json", "stream": False},
                timeout=18.0,
            )
            return json.loads(resp.json().get("response", "{}"))
        except Exception:
            return None

    def _synthesize_from_food_science(self, target: str, dish: str, wiki_text: str, off_text: str) -> list[dict]:
        """Rule-grounded food-science fallback driven by live Wikipedia/OpenFoodFacts composition."""
        combined = f"{target} {wiki_text} {off_text}".lower()
        slug = re.sub(r"[^a-z0-9]+", "_", target.lower())[:40]
        dish_list = [dish.lower()] if dish else ["savory snacks", "daily cooking"]

        # 1. Fresh Herb / Chutney / Dip / Condiment profile
        if any(k in combined for k in ("chutney", "coriander", "cilantro", "mint", "dip", "condiment", "relish", "sauce")):
            return [
                {
                    "id": f"live_sub_{slug}_1",
                    "target": target.title(),
                    "substitute": "Tamarind-Date (Imli) Chutney or Mint-Yogurt Raita Dip (3 tbsp yogurt + 1 tsp dried mint + pinch roasted cumin & lemon)",
                    "tastes": ["sour", "sweet", "cooling", "spicy_pungent"],
                    "cuisine": "Indian & South Asian",
                    "role": "Provides tart fruit/lactic acidity and fresh herbal aromatics to cut fried pastry richness",
                    "flavor_profile": "Tangy-sweet or cool herbaceous lift with mild spice warmth",
                    "compounds": ["tartaric acid", "lactic acid", "menthol", "linalool"],
                    "best_in": list(dict.fromkeys(dish_list + ["samosa", "pakora", "chaat", "kebabs"])),
                    "avoid_in": ["sweet desserts"],
                    "vegan": True,
                },
                {
                    "id": f"live_sub_{slug}_2",
                    "target": target.title(),
                    "substitute": "Cilantro-Lime Chimichurri / Salsa Verde (finely chopped parsley/cilantro + lime juice + green chili + garlic + olive oil)",
                    "tastes": ["sour", "cooling", "spicy_pungent"],
                    "cuisine": "Universal Pantry",
                    "role": "Fresh chlorophyll herb paste emulsified with citric acid and capsaicin heat",
                    "flavor_profile": "Bright, zesty, garlicky green herb sharpness",
                    "compounds": ["citric acid", "linalool", "capsaicin", "allicin"],
                    "best_in": list(dict.fromkeys(dish_list + ["samosa", "tacos", "grilled snacks"])),
                    "avoid_in": ["creamy sweet dishes"],
                    "vegan": True,
                },
            ]

        # 2. Spice / Herb / Seasoning profile
        if any(k in combined for k in ("spice", "herb", "seed", "powder", "masala", "seasoning", "leaf", "leaves")):
            role_desc = wiki_text[:200] if wiki_text else f"Aromatic seasoning profile matching {target}"
            return [
                {
                    "id": f"live_sub_{slug}_1",
                    "target": target.title(),
                    "substitute": f"Balanced aromatic blend matching {target} (1:1 ratio adjusted to taste)",
                    "tastes": ["spicy_pungent", "bitter"],
                    "cuisine": "Live Web Atlas",
                    "role": role_desc,
                    "flavor_profile": f"Warm aromatic profile derived from {target}",
                    "compounds": ["volatile terpenes", "essential oils"],
                    "best_in": dish_list,
                    "avoid_in": ["unspiced delicate pastries"],
                    "vegan": True,
                }
            ]

        # 3. General Ingredient fallback grounded strictly in live extract
        role_desc = wiki_text[:220] if wiki_text else f"Functional culinary equivalent for {target}"
        return [
            {
                "id": f"live_sub_{slug}_1",
                "target": target.title(),
                "substitute": f"Culinary functional equivalent for {target} (match acidity, fat, and texture)",
                "tastes": ["umami", "sour"],
                "cuisine": "Live Web Atlas",
                "role": role_desc,
                "flavor_profile": f"Matched to {target} composition",
                "compounds": ["organic acids", "volatile aroma compounds"],
                "best_in": dish_list,
                "avoid_in": ["incompatible flavor profiles"],
                "vegan": True,
            }
        ]

    def fetch_live_substitution(self, query: str, dish_context: str | None = None) -> list[dict]:
        target, dish = parse_substitution_intent(query, dish_context)
        wiki_title, wiki_text = self._wiki_extract_best(f"{target} culinary ingredient")
        off_info = self._openfoodfacts_profile(target)
        off_text = off_info.get("ingredients_text", "")

        llm_prompt = (
            "You are a food scientist. Return a JSON object with a single key 'substitutes' containing a list of "
            "2 practical ingredient/condiment substitutes for the target below. "
            "CRITICAL RULE: Never suggest a whole meal, bread, or pastry (such as dosa, roti, naan, rice, or samosa) "
            "as a substitute for a condiment, spice, or ingredient. Give exact ratios and chemical/culinary reasons.\n"
            "Each item must have keys: 'substitute' (string), 'tastes' (list of strings), 'cuisine' (string), "
            "'role' (string), 'flavor_profile' (string), 'compounds' (list of strings), 'best_in' (list of strings), "
            "'avoid_in' (list of strings), 'vegan' (bool).\n\n"
            f"Target to replace: {target}\n"
            f"Dish context: {dish or 'general cooking'}\n"
            f"Reference composition ({wiki_title}): {wiki_text}\n"
            f"Ingredient label data: {off_text}"
        )

        structured = self._try_ollama_structure(llm_prompt)
        cards = []
        slug = re.sub(r"[^a-z0-9]+", "_", target.lower())[:40]

        if structured and isinstance(structured.get("substitutes"), list):
            for idx, item in enumerate(structured["substitutes"][:2]):
                sub_text = str(item.get("substitute", "")).strip()
                if not sub_text or sub_text.lower() in DISALLOWED_SUBSTITUTES:
                    continue
                cards.append(
                    {
                        "id": f"live_sub_{slug}_{idx}",
                        "target": target.title(),
                        "substitute": sub_text,
                        "tastes": item.get("tastes") or ["sour", "umami"],
                        "cuisine": item.get("cuisine") or "Live Web Atlas",
                        "role": item.get("role") or "Chemically matched flavor and texture substitute",
                        "flavor_profile": item.get("flavor_profile") or "Balanced acidity and aroma",
                        "compounds": item.get("compounds") or ["organic acids", "volatile terpenes"],
                        "best_in": item.get("best_in") or ([dish] if dish else [target]),
                        "avoid_in": item.get("avoid_in") or ["incompatible flavor profiles"],
                        "vegan": bool(item.get("vegan", off_info.get("vegan", True))),
                    }
                )

        if not cards:
            cards = self._synthesize_from_food_science(target, dish, wiki_text, off_text)

        return cards

    def fetch_live_pairing(self, ingredient: str) -> dict | None:
        ing = ingredient.strip().lower()
        co_ings, dish_names = self._mealdb_cooccurrences(ing)
        _, wiki_text = self._wiki_extract_best(f"{ing} food ingredient")

        if not co_ings and not wiki_text:
            return None

        return {
            "ingredient": ing,
            "tastes": ["umami", "spicy_pungent"],
            "cuisine": "Live Web Atlas",
            "partners": co_ings or ["garlic", "lemon", "ginger", "cumin", "olive oil"],
            "compounds": ["volatile aroma compounds"],
            "dishes": dish_names or [f"{ing} preparation"],
            "summary": wiki_text[:240] if wiki_text else f"Live culinary profile for {ing}",
        }
