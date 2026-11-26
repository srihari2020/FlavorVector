import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Core food-science flavor graph (volatile compounds + culinary co-occurrence)
FOOD_SCIENCE_SEED = [
    {
        "ingredient": "rosemary",
        "partners": ["garlic", "lemon", "thyme", "olive oil", "chicken", "lamb", "potatoes", "black pepper"],
        "compounds": ["1,8-cineole", "alpha-pinene", "camphor", "borneol"],
        "dishes": ["roasted chicken", "herb focaccia", "garlic roasted potatoes", "grilled lamb chops"],
    },
    {
        "ingredient": "butter",
        "partners": ["garlic", "lemon", "thyme", "parsley", "shallot", "chicken", "black pepper", "white wine"],
        "compounds": ["diacetyl", "butyric acid", "delta-decalactone", "acetoin"],
        "dishes": ["pan-roasted chicken", "lemon butter sauce", "garlic herb basting", "mashed potatoes"],
    },
    {
        "ingredient": "olive oil",
        "partners": ["garlic", "lemon", "thyme", "rosemary", "basil", "tomato", "black pepper", "oregano"],
        "compounds": ["hexanal", "oleocanthal", "linalool", "squalene"],
        "dishes": ["Mediterranean roasted chicken", "herb marinade", "vinaigrette", "roasted vegetables"],
    },
    {
        "ingredient": "black pepper",
        "partners": ["garlic", "lemon", "thyme", "butter", "chicken", "olive oil", "parmesan", "rosemary"],
        "compounds": ["piperine", "beta-caryophyllene", "limonene", "sabinene"],
        "dishes": ["roasted chicken", "cacio e pepe", "steak au poivre", "lemon pepper wings"],
    },
    {
        "ingredient": "oregano",
        "partners": ["garlic", "lemon", "olive oil", "thyme", "tomato", "chicken", "basil", "feta"],
        "compounds": ["carvacrol", "thymol", "p-cymene", "gamma-terpinene"],
        "dishes": ["Greek lemon chicken", "marinara sauce", "grilled souvlaki", "roasted vegetables"],
    },
    {
        "ingredient": "parsley",
        "partners": ["garlic", "lemon", "butter", "olive oil", "thyme", "chicken", "shallot", "capers"],
        "compounds": ["myristicin", "apiol", "beta-phellandrene", "limonene"],
        "dishes": ["gremolata", "chimichurri", "herb roasted chicken", "garlic butter shrimp"],
    },
    {
        "ingredient": "shallot",
        "partners": ["garlic", "thyme", "butter", "white wine", "lemon", "chicken", "mushroom", "dijon mustard"],
        "compounds": ["dipropyl disulfide", "quercetin", "allyl propyl disulfide"],
        "dishes": ["pan sauce", "French vinaigrette", "braised chicken", "mushroom risotto"],
    },
    {
        "ingredient": "white wine",
        "partners": ["garlic", "lemon", "thyme", "butter", "shallot", "chicken", "parsley", "olive oil"],
        "compounds": ["ethyl octanoate", "tartaric acid", "linalool", "isoamyl acetate"],
        "dishes": ["chicken piccata", "white wine pan sauce", "risotto", "steamed mussels"],
    },
    {
        "ingredient": "dijon mustard",
        "partners": ["garlic", "lemon", "thyme", "honey", "olive oil", "chicken", "shallot", "white wine"],
        "compounds": ["allyl isothiocyanate", "sinigrin", "acetic acid"],
        "dishes": ["honey mustard roasted chicken", "vinaigrette", "herb crusted pork", "pan sauce"],
    },
    {
        "ingredient": "honey",
        "partners": ["lemon", "garlic", "thyme", "dijon mustard", "soy sauce", "ginger", "chicken", "chili"],
        "compounds": ["phenylacetic acid", "linalool", "furfural", "benzaldehyde"],
        "dishes": ["honey lemon glazed chicken", "roasted carrots", "glaze", "vinaigrette"],
    },
    {
        "ingredient": "smoked paprika",
        "partners": ["garlic", "chicken", "lemon", "olive oil", "onion", "cumin", "thyme", "tomato"],
        "compounds": ["capsanthin", "pyrazines", "beta-ionone", "guaiacol"],
        "dishes": ["Spanish roasted chicken", "patatas bravas", "paella", "dry rub"],
    },
    {
        "ingredient": "garlic",
        "partners": ["lemon", "thyme", "butter", "olive oil", "rosemary", "ginger", "onion", "black pepper"],
        "compounds": ["allicin", "diallyl disulfide", "allyl methyl sulfide"],
        "dishes": ["roasted chicken", "stir-fry", "pasta aglio e olio", "herb marinades"],
    },
    {
        "ingredient": "lemon",
        "partners": ["garlic", "thyme", "butter", "olive oil", "rosemary", "parsley", "chicken", "honey"],
        "compounds": ["limonene", "citral", "citric acid", "gamma-terpinene"],
        "dishes": ["lemon herb roasted chicken", "piccata", "ceviche", "lemon tart"],
    },
    {
        "ingredient": "thyme",
        "partners": ["garlic", "lemon", "butter", "rosemary", "chicken", "mushroom", "shallot", "white wine"],
        "compounds": ["thymol", "carvacrol", "p-cymene", "linalool"],
        "dishes": ["roasted chicken", "beef bourguignon", "mushroom soup", "roasted root vegetables"],
    },
    {
        "ingredient": "ginger",
        "partners": ["garlic", "soy sauce", "scallion", "sesame oil", "chili", "lime", "honey", "chicken"],
        "compounds": ["gingerol", "zingerone", "citral", "beta-sesquiphellandrene"],
        "dishes": ["ginger scallion chicken", "stir-fry", "ramen broth", "curry"],
    },
    {
        "ingredient": "soy sauce",
        "partners": ["ginger", "garlic", "sesame oil", "scallion", "honey", "rice vinegar", "chili", "mushroom"],
        "compounds": ["glutamic acid", "methional", "4-hydroxy-2,5-dimethyl-3(2H)-furanone"],
        "dishes": ["teriyaki glaze", "stir-fry", "fried rice", "umami marinade"],
    },
    {
        "ingredient": "sesame oil",
        "partners": ["soy sauce", "ginger", "garlic", "scallion", "rice vinegar", "chili", "chicken", "tofu"],
        "compounds": ["2-furfurylthiol", "pyrazines", "sesamol", "guaiacol"],
        "dishes": ["sesame noodles", "fried rice", "dumpling dipping sauce", "stir-fry"],
    },
    {
        "ingredient": "cumin",
        "partners": ["coriander", "garlic", "turmeric", "chili", "onion", "lime", "chickpeas", "lamb"],
        "compounds": ["cuminaldehyde", "gamma-terpinene", "beta-pinene"],
        "dishes": ["spiced roasted cauliflower", "tacos", "dal", "moroccan stew"],
    },
    {
        "ingredient": "coriander",
        "partners": ["cumin", "turmeric", "garlic", "ginger", "lime", "chili", "cilantro", "chicken"],
        "compounds": ["linalool", "alpha-pinene", "geranyl acetate"],
        "dishes": ["curry", "spiced rubs", "falafel", "roasted carrots"],
    },
    {
        "ingredient": "turmeric",
        "partners": ["cumin", "coriander", "ginger", "garlic", "black pepper", "coconut milk", "cauliflower"],
        "compounds": ["curcumin", "ar-turmerone", "zingiberene"],
        "dishes": ["golden curry", "roasted cauliflower", "dal tadka", "spiced rice"],
    },
    {
        "ingredient": "basil",
        "partners": ["tomato", "garlic", "olive oil", "mozzarella", "parmesan", "lemon", "pine nuts"],
        "compounds": ["estragole", "linalool", "eugenol", "1,8-cineole"],
        "dishes": ["pesto", "caprese salad", "margherita pizza", "tomato pasta"],
    },
    {
        "ingredient": "tomato",
        "partners": ["basil", "garlic", "olive oil", "oregano", "onion", "parmesan", "balsamic vinegar"],
        "compounds": ["glutamic acid", "hexanal", "beta-ionone", "furaneol"],
        "dishes": ["marinara", "shakshuka", "bruschetta", "ragu"],
    },
    {
        "ingredient": "mushroom",
        "partners": ["thyme", "garlic", "butter", "shallot", "soy sauce", "parmesan", "white wine", "black pepper"],
        "compounds": ["1-octen-3-ol", "guanylate", "glutamic acid", "methional"],
        "dishes": ["mushroom risotto", "stroganoff", "roasted mushrooms", "umami broth"],
    },
    {
        "ingredient": "parmesan",
        "partners": ["garlic", "basil", "black pepper", "butter", "olive oil", "mushroom", "lemon", "tomato"],
        "compounds": ["butyric acid", "glutamic acid", "ethyl hexanoate", "isovaleric acid"],
        "dishes": ["risotto", "pesto", "caesar salad", "pasta"],
    },
    {
        "ingredient": "dark chocolate",
        "partners": ["espresso", "vanilla", "sea salt", "cinnamon", "chili", "orange", "hazelnut", "butter"],
        "compounds": ["pyrazines", "theobromine", "isovaleraldehyde", "phenylethyl alcohol"],
        "dishes": ["mocha mousse", "mole sauce", "brownies", "chocolate ganache"],
    },
    {
        "ingredient": "espresso",
        "partners": ["dark chocolate", "vanilla", "cinnamon", "cardamom", "butter", "brown sugar", "cream"],
        "compounds": ["2-furfurylthiol", "pyrazines", "guaiacol", "vanillin"],
        "dishes": ["tiramisu", "mocha brownies", "coffee rub for steak", "affogato"],
    },
    {
        "ingredient": "cinnamon",
        "partners": ["apple", "cardamom", "vanilla", "nutmeg", "clove", "honey", "dark chocolate", "cumin"],
        "compounds": ["cinnamaldehyde", "eugenol", "linalool"],
        "dishes": ["apple pie", "chai", "moroccan tagine", "cinnamon rolls"],
    },
    {
        "ingredient": "cardamom",
        "partners": ["cinnamon", "ginger", "clove", "vanilla", "orange", "espresso", "rice", "pistachio"],
        "compounds": ["1,8-cineole", "alpha-terpinyl acetate", "linalool"],
        "dishes": ["masala chai", "biryani", "kheer", "spiced buns"],
    },
    {
        "ingredient": "lime",
        "partners": ["cilantro", "chili", "garlic", "ginger", "cumin", "fish sauce", "coconut milk", "avocado"],
        "compounds": ["limonene", "citral", "citric acid", "beta-pinene"],
        "dishes": ["guacamole", "ceviche", "pad thai", "tacos"],
    },
    {
        "ingredient": "coconut milk",
        "partners": ["lime", "ginger", "lemongrass", "turmeric", "chili", "garlic", "curry leaves", "cilantro"],
        "compounds": ["delta-octalactone", "gamma-nonalactone", "octanoic acid"],
        "dishes": ["Thai green curry", "laksa", "coconut rice", "Malabar stew"],
    },
]


def load_recipe_cooccurrence(recipes_path: Path) -> tuple[Counter, dict[str, Counter], dict[str, list[str]]]:
    freq: Counter[str] = Counter()
    cooccur: dict[str, Counter[str]] = defaultdict(Counter)
    sample_dishes: dict[str, list[str]] = defaultdict(list)

    if not recipes_path.exists():
        return freq, cooccur, sample_dishes

    with open(recipes_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                recipe = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (recipe.get("title") or "").strip()
            raw_items = recipe.get("ner") or recipe.get("ingredients") or []
            if not isinstance(raw_items, list):
                continue
            items = sorted({str(x).strip().lower() for x in raw_items if 2 <= len(str(x).strip()) <= 35})
            for item in items:
                freq[item] += 1
                if title and len(sample_dishes[item]) < 5 and title not in sample_dishes[item]:
                    sample_dishes[item].append(title)
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    cooccur[a][b] += 1
                    cooccur[b][a] += 1

    return freq, cooccur, sample_dishes


def build_profiles(top_n: int = 1500) -> list[dict]:
    freq, cooccur, sample_dishes = load_recipe_cooccurrence(RAW_DIR / "recipes_25k.jsonl")
    merged: dict[str, dict] = {}

    # 1. Load curated food-science seed first
    for entry in FOOD_SCIENCE_SEED:
        ing = entry["ingredient"].lower()
        merged[ing] = {
            "ingredient": ing,
            "frequency": max(freq.get(ing, 0), 250),
            "partners": list(entry["partners"]),
            "compounds": list(entry["compounds"]),
            "dishes": list(entry["dishes"]),
        }

    # 2. Merge mined recipe co-occurrences if recipes_25k.jsonl had valid entries
    for ing, count in freq.most_common(top_n):
        partners = [p for p, _ in cooccur[ing].most_common(10)]
        dishes = sample_dishes[ing]
        if ing in merged:
            existing_partners = merged[ing]["partners"]
            for p in partners:
                if p not in existing_partners and len(existing_partners) < 12:
                    existing_partners.append(p)
        else:
            merged[ing] = {
                "ingredient": ing,
                "frequency": count,
                "partners": partners,
                "compounds": [],
                "dishes": dishes,
            }

    profiles = []
    for idx, (ing, data) in enumerate(merged.items()):
        lines = [
            f"Ingredient: {ing}",
            f"Top Culinary Pairing Partners: {', '.join(data['partners'])}",
            f"Best Dish Contexts: {', '.join(data['dishes'])}",
        ]
        if data["compounds"]:
            lines.append(f"Key Volatile & Flavor Compounds: {', '.join(data['compounds'])}")

        profiles.append(
            {
                "id": f"ing_{idx}_{re.sub(r'[^a-z0-9]+', '_', ing)[:40]}",
                "ingredient": ing,
                "frequency": data["frequency"],
                "partners": data["partners"],
                "compounds": data["compounds"],
                "document": "\n".join(lines),
            }
        )
    return profiles


def main() -> None:
    profiles = build_profiles(top_n=1500)
    print(f"Prepared {len(profiles):,} ingredient flavor profiles. Loading {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="flavor_pairings",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [p["id"] for p in profiles]
    docs = [p["document"] for p in profiles]
    metas = [
        {
            "ingredient": p["ingredient"],
            "frequency": p["frequency"],
            "partners": ", ".join(p["partners"][:8]),
            "compounds": ", ".join(p["compounds"][:6]),
        }
        for p in profiles
    ]

    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Success! Total vectors in flavor_pairings: {collection.count()}")


if __name__ == "__main__":
    main()
