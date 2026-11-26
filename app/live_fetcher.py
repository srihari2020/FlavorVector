import json
import re
import requests

HEADERS = {"User-Agent": "FlavorVector/1.0 (Food-Science Retrieval Engine)"}

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


def _ensure_str_list(val, default: list[str]) -> list[str]:
    if isinstance(val, list):
        cleaned = [str(x).strip() for x in val if str(x).strip()]
        return cleaned or default
    if isinstance(val, str) and val.strip():
        cleaned = [x.strip() for x in val.split(",") if x.strip()]
        return cleaned or default
    return default


class LiveCulinaryFetcher:
    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout

    def _wiki_extract_best(self, query: str) -> tuple[str, str]:
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
                timeout=20.0,
            )
            parsed = json.loads(resp.json().get("response", "{}"))
            if isinstance(parsed, list):
                return {"substitutes": parsed}
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None

    def _synthesize_from_food_science(self, target: str, dish: str, wiki_text: str, off_text: str) -> list[dict]:
        combined = f"{target} {dish} {wiki_text} {off_text}".lower()
        slug = re.sub(r"[^a-z0-9]+", "_", target.lower())[:40]
        dish_list = [dish.lower()] if dish else ["savory snacks", "daily cooking"]

        # 1. Pickling / Sour Fruit / Raw Mango profile
        if any(k in combined for k in ("pickle", "achar", "avakaya", "raw mango", "unripe mango", "mango")):
            return [
                {
                    "id": f"live_sub_{slug}_1",
                    "target": target.title(),
                    "substitute": "Unripe Green Papaya or Raw Gooseberry (Amla) — 1:1 diced ratio + 1 tbsp lemon juice",
                    "tastes": ["sour", "astringent"],
                    "cuisine": "Indian & South Asian",
                    "role": "Firm pectin-rich flesh holds crunch in spiced oil/brine while supplying sharp tartness",
                    "flavor_profile": "Crisp, tangy, mildly astringent fruit acidity",
                    "compounds": ["citric acid", "malic acid", "pectin", "ascorbic acid"],
                    "best_in": list(dict.fromkeys(dish_list + ["pickle", "achar", "chutney", "relishes"])),
                    "avoid_in": ["sweet ripe mango desserts"],
                    "vegan": True,
                },
                {
                    "id": f"live_sub_{slug}_2",
                    "target": target.title(),
                    "substitute": "Unripe Green Tomatoes or Hog Plum (Ambarella / Amtekai) — 1:1 ratio",
                    "tastes": ["sour", "umami"],
                    "cuisine": "Indian & South Asian",
                    "role": "High malic/citric acid and firm cell walls resist turning mushy under mustard-chili tempering",
                    "flavor_profile": "Bright, tart, vegetal crunch",
                    "compounds": ["malic acid", "citric acid", "glutamic acid"],
                    "best_in": list(dict.fromkeys(dish_list + ["instant pickle", "thokku", "chutney"])),
                    "avoid_in": ["milkshakes"],
                    "vegan": True,
                },
            ]

        # 2. Fresh Herb / Chutney / Dip / Condiment profile
        if any(k in combined for k in ("chutney", "coriander", "cilantro", "mint", "dip", "condiment", "relish", "sauce")):
            return [
                {
                    "id": f"live_sub_{slug}_1",
                    "target": target.title(),
                    "substitute": "Tamarind-Date (Imli) Chutney or Mint-Yogurt Raita Dip",
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
                    "substitute": "Cilantro-Lime Chimichurri / Salsa Verde (parsley/cilantro + lime juice + green chili + garlic + olive oil)",
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

        role_desc = wiki_text[:220] if wiki_text else f"Functional culinary equivalent for {target}"
        return [
            {
                "id": f"live_sub_{slug}_1",
                "target": target.title(),
                "substitute": f"Culinary functional equivalent for {target} (match acidity, texture, and aroma)",
                "tastes": ["sour", "umami"],
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
            "You are a food scientist. Return ONLY valid JSON matching this exact structure:\n"
            '{"substitutes": [{"substitute": "1 cup ingredient + ratio", "tastes": ["sour", "astringent"], '
            '"cuisine": "Indian & South Asian", "role": "Chemical and textural role", '
            '"flavor_profile": "Taste and aroma description", "compounds": ["citric acid", "pectin"], '
            '"best_in": ["pickle", "chutney"], "avoid_in": ["desserts"], "vegan": true}]}\n'
            "Rules: Provide 2 practical ingredient substitutes with exact ratios. "
            "Never suggest whole meals or breads (dosa, roti, naan, rice, samosa) as substitutes.\n\n"
            f"Target to replace: {target}\n"
            f"Dish context: {dish or 'general cooking'}\n"
            f"Reference ({wiki_title}): {wiki_text}\n"
            f"Label info: {off_text}"
        )

        structured = self._try_ollama_structure(llm_prompt)
        cards = []
        slug = re.sub(r"[^a-z0-9]+", "_", target.lower())[:40]

        raw_subs = structured.get("substitutes") if isinstance(structured, dict) else None
        if isinstance(raw_subs, dict):
            raw_subs = [raw_subs]

        if isinstance(raw_subs, list):
            for idx, item in enumerate(raw_subs[:2]):
                if isinstance(item, str):
                    sub_text = item.strip()
                    if not sub_text or sub_text.lower() in DISALLOWED_SUBSTITUTES:
                        continue
                    cards.append(
                        {
                            "id": f"live_sub_{slug}_{idx}",
                            "target": target.title(),
                            "substitute": sub_text,
                            "tastes": ["sour", "umami"],
                            "cuisine": "Live Web Atlas",
                            "role": f"Functional substitute for {target} in {dish or 'cooking'}",
                            "flavor_profile": "Matched acidity, texture, and flavor balance",
                            "compounds": ["organic acids", "volatile terpenes"],
                            "best_in": [dish.lower()] if dish else [target.lower()],
                            "avoid_in": ["incompatible flavor profiles"],
                            "vegan": bool(off_info.get("vegan", True)),
                        }
                    )
                elif isinstance(item, dict):
                    sub_text = str(item.get("substitute", "")).strip()
                    if not sub_text or sub_text.lower() in DISALLOWED_SUBSTITUTES:
                        continue
                    cards.append(
                        {
                            "id": f"live_sub_{slug}_{idx}",
                            "target": target.title(),
                            "substitute": sub_text,
                            "tastes": _ensure_str_list(item.get("tastes"), ["sour", "umami"]),
                            "cuisine": str(item.get("cuisine") or "Live Web Atlas"),
                            "role": str(item.get("role") or "Chemically matched flavor and texture substitute"),
                            "flavor_profile": str(item.get("flavor_profile") or "Balanced acidity and aroma"),
                            "compounds": _ensure_str_list(item.get("compounds"), ["organic acids", "volatile terpenes"]),
                            "best_in": _ensure_str_list(item.get("best_in"), [dish.lower()] if dish else [target.lower()]),
                            "avoid_in": _ensure_str_list(item.get("avoid_in"), ["incompatible flavor profiles"]),
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
