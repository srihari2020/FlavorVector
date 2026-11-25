import json
import zipfile
from pathlib import Path
import requests
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_DATASETS = [
    ("mbien/recipe_nlg", "refs/convert/parquet"),
    ("GPTasty/PAID-recipes-clean", None),
    ("Shengtao/recipe", None),
    ("corbt/all-recipes", None),
]


def get_recipe_stream():
    for repo_id, revision in CANDIDATE_DATASETS:
        try:
            print(f"Trying {repo_id} (revision={revision or 'default'})...")
            ds = load_dataset(
                repo_id,
                revision=revision,
                split="train",
                streaming=True,
            )
            iterator = iter(ds)
            first_row = next(iterator)
            print(f"Connected to {repo_id} successfully.")
            return repo_id, first_row, iterator
        except Exception as exc:
            print(f"Skipping {repo_id}: {exc}")
    raise RuntimeError("Could not connect to any Parquet recipe dataset on Hugging Face.")


def normalize_row(idx: int, row: dict) -> dict:
    return {
        "id": str(row.get("id", idx)),
        "title": row.get("title") or row.get("name") or f"Recipe {idx}",
        "ingredients": row.get("ingredients") or row.get("raw_ingredients") or row.get("input") or [],
        "ner": row.get("ner") or row.get("NER") or [],
    }


def fetch_recipes(limit: int = 25_000) -> None:
    out_path = RAW_DIR / "recipes_25k.jsonl"
    _, first_row, iterator = get_recipe_stream()

    print(f"Streaming {limit} recipes to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(normalize_row(0, first_row)) + "\n")
        for i, row in enumerate(iterator, start=1):
            if i >= limit:
                break
            f.write(json.dumps(normalize_row(i, row)) + "\n")
            if i % 5_000 == 0:
                print(f"  Saved {i:,} / {limit:,} recipes...")

    print(f"Completed: saved {limit:,} recipes to {out_path}")


def fetch_foodb() -> None:
    url = "https://foodb.ca/public/system/downloads/foodb_2020_04_07_json.zip"
    zip_path = RAW_DIR / "foodb_json.zip"
    extract_dir = RAW_DIR / "foodb"

    print("Downloading FooDB JSON dataset (86.6 MB)...")
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=60,
            headers={"User-Agent": "FlavorVector/1.0"},
        )
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted FooDB JSON tables to {extract_dir}")
    except Exception as exc:
        print(f"FooDB download skipped ({exc}). You can manually grab it from https://foodb.ca/downloads")


def write_substitution_seed() -> None:
    seed_rules = [
        {
            "id": "sub_buttermilk_lemon",
            "target": "Buttermilk",
            "substitute": "1 cup whole milk + 1 tbsp fresh lemon juice (rest 10 min)",
            "role": "Acidic liquid, protein tenderizer, activates baking soda for leavening",
            "flavor_profile": "Tangy, bright citrus acidity, mild dairy creaminess",
            "compounds": ["citric acid", "lactic acid", "milk triglycerides"],
            "best_in": ["pancakes", "waffles", "biscuits", "quick breads"],
            "avoid_in": ["thick salad dressings without extra emulsifier"],
            "vegan": False,
        },
        {
            "id": "sub_buttermilk_yogurt",
            "target": "Buttermilk",
            "substitute": "3/4 cup plain yogurt + 1/4 cup water",
            "role": "Lactic acid source for baking soda leavening plus cultured viscosity",
            "flavor_profile": "Rich cultured tang, creamy dairy finish",
            "compounds": ["lactic acid", "diacetyl", "acetaldehyde"],
            "best_in": ["pancakes", "muffins", "cakes", "marinades"],
            "avoid_in": ["thin batters where viscosity is not adjusted"],
            "vegan": False,
        },
        {
            "id": "sub_buttermilk_vegan",
            "target": "Buttermilk",
            "substitute": "1 cup unsweetened soy milk + 1 tbsp apple cider vinegar",
            "role": "Curdles soy protein to mimic dairy body and provides acid for baking soda",
            "flavor_profile": "Sharp tang, neutral-nutty base",
            "compounds": ["acetic acid", "soy proteins"],
            "best_in": ["vegan pancakes", "cornbread", "quick breads"],
            "avoid_in": ["delicate custards"],
            "vegan": True,
        },
        {
            "id": "sub_egg_leavener",
            "target": "1 Large Egg (Leavening & Lift)",
            "substitute": "1 tsp baking soda + 1 tbsp white or apple cider vinegar",
            "role": "Rapid CO2 gas release for airy crumb structure",
            "flavor_profile": "Neutral",
            "compounds": ["sodium bicarbonate", "acetic acid"],
            "best_in": ["cupcakes", "light cakes", "quick breads"],
            "avoid_in": ["meatballs", "dense brownies", "custards"],
            "vegan": True,
        },
        {
            "id": "sub_egg_binder",
            "target": "1 Large Egg (Moisture & Binding)",
            "substitute": "1 tbsp ground flaxseed + 3 tbsp warm water (rest 10 min)",
            "role": "Mucilage gel network that binds particles and retains moisture",
            "flavor_profile": "Mild, earthy, slightly nutty",
            "compounds": ["flax mucilage polysaccharides", "alpha-linolenic acid"],
            "best_in": ["meatballs", "veggie burgers", "oatmeal cookies", "pancakes"],
            "avoid_in": ["meringues", "angel food cake", "souffles"],
            "vegan": True,
        },
        {
            "id": "sub_fish_sauce_umami",
            "target": "1 tbsp Fish Sauce",
            "substitute": "1 tbsp soy sauce + 1/4 tsp rice vinegar + pinch of crushed nori or mushroom powder",
            "role": "Deep glutamate-ribonucleotide umami synergy and salinity",
            "flavor_profile": "Savory, oceanic umami, balanced salt-acid",
            "compounds": ["glutamic acid", "inosinate", "guanylate"],
            "best_in": ["stir-fries", "broths", "curries", "dipping sauces"],
            "avoid_in": ["raw dishes requiring fermented anchovy aroma"],
            "vegan": True,
        },
    ]
    out_path = RAW_DIR / "substitutions_seed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(seed_rules, f, indent=2)
    print(f"Wrote {len(seed_rules)} food-science rules to {out_path}")


if __name__ == "__main__":
    fetch_recipes(limit=25_000)
    fetch_foodb()
    write_substitution_seed()
