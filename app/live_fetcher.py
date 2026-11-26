import json
import re
import requests

HEADERS = {"User-Agent": "FlavorVector/1.0 (Local Culinary Search Engine)"}


def parse_substitution_intent(query: str, dish_context: str | None = None) -> tuple[str, str]:
    """Extract target ingredient and dish context from natural-language queries."""
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

    def _wiki_search(self, term: str, limit: int = 3) -> list[dict]:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{term} food cooking",
            "srlimit": limit,
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            results = []
            for h in hits:
                title = h.get("title", "")
                snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                extract = self._wiki_extract(title)
                results.append({"title": title, "summary": extract or snippet})
            return results
        except Exception:
            return []

    def _wiki_extract(self, title: str) -> str:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=self.timeout)
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = (page.get("extract") or "").strip()
                if text:
                    return text[:500]
        except Exception:
            pass
        return ""

    def _mealdb_recipes(self, keyword: str) -> list[dict]:
        meals = []
        for endpoint in (
            f"https://www.themealdb.com/api/json/v1/1/search.php?s={requests.utils.quote(keyword)}",
            f"https://www.themealdb.com/api/json/v1/1/filter.php?i={requests.utils.quote(keyword)}",
        ):
            try:
                resp = requests.get(endpoint, headers=HEADERS, timeout=self.timeout)
                data = resp.json().get("meals") or []
                for m in data[:5]:
                    ings = [
                        m.get(f"strIngredient{i}", "").strip().lower()
                        for i in range(1, 21)
                        if m.get(f"strIngredient{i}") and m.get(f"strIngredient{i}").strip()
                    ]
                    meals.append(
                        {
                            "name": m.get("strMeal", ""),
                            "category": m.get("strCategory", ""),
                            "area": m.get("strArea", ""),
                            "ingredients": ings,
                        }
                    )
            except Exception:
                continue
        return meals

    def _openfoodfacts_check(self, term: str) -> dict:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {"search_terms": term, "search_simple": 1, "action": "process", "json": 1, "page_size": 3}
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=self.timeout)
            products = resp.json().get("products") or []
            if not products:
                return {}
            p = products[0]
            labels = [lbl.lower() for lbl in (p.get("labels_tags") or [])]
            is_vegan = any("vegan" in lbl for lbl in labels) or "en:non-vegan" not in labels
            return {
                "product_name": p.get("product_name", term),
                "ingredients_text": (p.get("ingredients_text") or "")[:240],
                "vegan": is_vegan,
            }
        except Exception:
            return {}

    def _try_ollama_structure(self, prompt: str) -> dict | None:
        """Use local Ollama if running on localhost:11434; otherwise fall back cleanly."""
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
                timeout=15.0,
            )
            return json.loads(resp.json().get("response", "{}"))
        except Exception:
            return None

    def fetch_live_substitution(self, query: str, dish_context: str | None = None) -> list[dict]:
        target, dish = parse_substitution_intent(query, dish_context)
        wiki_target = self._wiki_search(target, limit=2)
        wiki_sub = self._wiki_search(f"{target} substitute alternative", limit=2)
        wiki_dish = self._wiki_search(dish, limit=2) if dish else []
        meals = self._mealdb_recipes(dish or target)
        off_info = self._openfoodfacts_check(target)

        context_lines = [f"Target: {target}", f"Dish: {dish or 'general cooking'}"]
        for item in (wiki_target + wiki_sub + wiki_dish)[:4]:
            context_lines.append(f"- {item['title']}: {item['summary']}")
        if meals:
            meal_names = ", ".join(m["name"] for m in meals[:4] if m["name"])
            context_lines.append(f"- Related recipes: {meal_names}")
        if off_info.get("ingredients_text"):
            context_lines.append(f"- Composition: {off_info['ingredients_text']}")

        raw_context = "\n".join(context_lines)

        # Optional local LLM structuring if Ollama is active
        llm_prompt = (
            "You are a food scientist. Based on the live culinary data below, return a JSON object with a single key "
            "'substitutes' containing a list of 2 practical substitutes. Each item must have keys: "
            "'substitute', 'tastes' (list), 'cuisine', 'role', 'flavor_profile', 'compounds' (list), "
            "'best_in' (list), 'avoid_in' (list), 'vegan' (bool).\n\n"
            f"{raw_context}"
        )
        structured = self._try_ollama_structure(llm_prompt)
        cards = []

        if structured and isinstance(structured.get("substitutes"), list) and structured["substitutes"]:
            for idx, item in enumerate(structured["substitutes"][:2]):
                slug = re.sub(r"[^a-z0-9]+", "_", f"{target}_{idx}")[:48]
                cards.append(
                    {
                        "id": f"live_sub_{slug}",
                        "target": f"{target.title()} ({dish.title()})" if dish else target.title(),
                        "substitute": item.get("substitute", "Live culinary alternative"),
                        "tastes": item.get("tastes") or ["umami", "sour"],
                        "cuisine": item.get("cuisine") or "Live Web Atlas",
                        "role": item.get("role") or "Functional flavor and texture match from live food-science sources",
                        "flavor_profile": item.get("flavor_profile") or "Balanced acidity, aroma, and mouthfeel",
                        "compounds": item.get("compounds") or ["organic acids", "volatile terpenes"],
                        "best_in": item.get("best_in") or ([dish] if dish else [target]),
                        "avoid_in": item.get("avoid_in") or ["incompatible flavor profiles"],
                        "vegan": bool(item.get("vegan", off_info.get("vegan", True))),
                    }
                )
        else:
            # Deterministic real-time extraction from live Wikipedia + TheMealDB + OpenFoodFacts
            related_titles = [
                w["title"]
                for w in (wiki_sub + wiki_dish + wiki_target)
                if w["title"].lower() != target.lower()
            ]
            alt_names = list(dict.fromkeys(related_titles))[:3]
            summary_snippet = (
                wiki_sub[0]["summary"][:220]
                if wiki_sub
                else (wiki_target[0]["summary"][:220] if wiki_target else f"Live culinary profile for {target}")
            )
            best_dishes = [dish] if dish else []
            best_dishes.extend(m["name"].lower() for m in meals[:3] if m.get("name"))
            if not best_dishes:
                best_dishes = ["savory snacks", "dipping sauces", "daily cooking"]

            sub_label = (
                f"{' or '.join(alt_names[:2])} (adjusted for {dish or target})"
                if alt_names
                else f"Tangy-spiced {target} base (lemon/tamarind + fresh herbs/spices)"
            )
            slug = re.sub(r"[^a-z0-9]+", "_", target)[:45]
            cards.append(
                {
                    "id": f"live_sub_{slug}",
                    "target": f"{target.title()} ({dish.title()})" if dish else target.title(),
                    "substitute": sub_label,
                    "tastes": ["sour", "umami", "spicy_pungent"],
                    "cuisine": "Live Web Atlas",
                    "role": summary_snippet,
                    "flavor_profile": f"Live-matched culinary profile for {target}",
                    "compounds": ["citric acid", "volatile aroma terpenes"],
                    "best_in": list(dict.fromkeys(best_dishes))[:5],
                    "avoid_in": ["sweet desserts unless balanced"],
                    "vegan": bool(off_info.get("vegan", True)),
                }
            )

        return cards

    def fetch_live_pairing(self, ingredient: str) -> dict | None:
        ing = ingredient.strip().lower()
        meals = self._mealdb_recipes(ing)
        wiki = self._wiki_search(ing, limit=1)

        co_items: list[str] = []
        dish_names: list[str] = []
        for m in meals:
            if m.get("name"):
                dish_names.append(m["name"].lower())
            for item in m.get("ingredients", []):
                if item != ing and item not in co_items and len(co_items) < 10:
                    co_items.append(item)

        if not co_items and not wiki:
            return None

        summary = wiki[0]["summary"][:240] if wiki else f"Live culinary profile for {ing}"
        return {
            "ingredient": ing,
            "tastes": ["umami", "spicy_pungent"],
            "cuisine": "Live Web Atlas",
            "partners": co_items or ["garlic", "lemon", "ginger", "cumin", "olive oil"],
            "compounds": ["volatile aroma compounds"],
            "dishes": dish_names[:5] or [f"{ing} preparation"],
            "summary": summary,
        }
