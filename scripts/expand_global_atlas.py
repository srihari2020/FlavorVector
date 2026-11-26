import re
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from build_universal_atlas import UNIVERSAL_SUBSTITUTIONS, UNIVERSAL_INGREDIENTS, merge_local_recipes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

EXTRA_SUBSTITUTIONS = [
    # --- SOUTH INDIAN CHUTNEYS, TIFFINS & ACCOMPANIMENTS ---
    {
        "id": "sub_coconut_chutney_peanut",
        "target": "Coconut Chutney (for Dosa, Idli, Vada & Uttapam)",
        "substitute": "Roasted Peanut (Groundnut) Chutney blended with roasted chana dal, green chili, ginger, tamarind & curry leaf tadka",
        "tastes": ["fatty_rich", "spicy_pungent", "umami", "kokumi"],
        "cuisine": "Indian & South Asian",
        "role": "Creamy roasted nut-legume lipid emulsion that clings to crisp dosa and balances fermented batter tang",
        "flavor_profile": "Nutty, rich, savory, mild chili-ginger warmth",
        "compounds": ["oleic acid", "pyrazines", "capsaicin", "gingerol"],
        "best_in": ["masala dosa", "idli", "medu vada", "uttapam", "pongal"],
        "avoid_in": ["sweet coconut desserts"],
        "vegan": True,
    },
    {
        "id": "sub_coconut_chutney_chana_dal",
        "target": "Coconut Chutney (No-Coconut White Tiffin Chutney)",
        "substitute": "Roasted Chana Dal (Pottukadalai / Dalia) Chutney blended with green chili, ginger, cilantro, lemon juice & mustard tadka",
        "tastes": ["fatty_rich", "cooling", "spicy_pungent"],
        "cuisine": "Indian & South Asian",
        "role": "Toasted split chickpea starches create the exact creamy spoon-coating consistency of coconut chutney",
        "flavor_profile": "Mild, creamy, fresh herbaceous-nutty finish",
        "compounds": ["amylose", "pyrazines", "linalool", "limonene"],
        "best_in": ["dosa", "idli", "rava dosa", "upma", "pesarattu"],
        "avoid_in": ["coconut milk curries"],
        "vegan": True,
    },
    {
        "id": "sub_coconut_chutney_kara",
        "target": "Coconut Chutney (Tangy-Spicy Dosa Accompaniment)",
        "substitute": "Tomato-Onion Kara Chutney (sauteed shallots, ripe tomatoes, dried red chilies, garlic & urad dal)",
        "tastes": ["umami", "sour", "spicy_pungent"],
        "cuisine": "Indian & South Asian",
        "role": "Glutamate-rich tomato-allium glaze that cuts oily crisp dosa and adds bright acidity",
        "flavor_profile": "Tangy, garlicky, savory roasted chili warmth",
        "compounds": ["glutamic acid", "lycopene", "capsaicin", "allicin"],
        "best_in": ["plain dosa", "set dosa", "idli", "paniyaram", "chapati"],
        "avoid_in": ["bland baby food"],
        "vegan": True,
    },
    {
        "id": "sub_coconut_chutney_sesame",
        "target": "Coconut Chutney (Nutty Sesame Alternative)",
        "substitute": "Roasted White Sesame (Til) & Mint-Coriander Chutney with tamarind and garlic",
        "tastes": ["fatty_rich", "bitter", "cooling", "sour"],
        "cuisine": "Indian & South Asian",
        "role": "Toasted sesame oils provide rich mouthfeel alongside bright herbal cooling notes",
        "flavor_profile": "Toasted nutty depth with fresh green herb lift",
        "compounds": ["sesamol", "pyrazines", "menthol", "linalool"],
        "best_in": ["dosa", "idli", "vada", "dhokla", "paratha"],
        "avoid_in": ["fruit salads"],
        "vegan": True,
    },
    {
        "id": "sub_sambar_tiffin_kurma",
        "target": "Sambar (Lentil Accompaniment for Dosa & Idli)",
        "substitute": "Vegetable Kurma (coconut-cashew-fennel gravy) or Moong Dal Tadka with tomato and lemon",
        "tastes": ["umami", "fatty_rich", "cooling"],
        "cuisine": "Indian & South Asian",
        "role": "Warm spiced liquid gravy that softens steamed idli and crisp dosa",
        "flavor_profile": "Aromatic, savory, mildly sweet spiced broth",
        "compounds": ["anethole", "1,8-cineole", "glutamic acid"],
        "best_in": ["idli", "dosa", "appam", "parotta"],
        "avoid_in": ["dry snacks"],
        "vegan": True,
    },
    {
        "id": "sub_dosa_batter_instant",
        "target": "Fermented Rice-Urad Dosa Batter",
        "substitute": "1 cup fine semolina (rava/sooji) + 1/2 cup rice flour + 1/2 cup sour yogurt + water (rest 20 min)",
        "tastes": ["sour", "fatty_rich"],
        "cuisine": "Indian & South Asian",
        "role": "Yogurt lactic acid mimics natural fermentation tang while rice flour crisps on hot griddle",
        "flavor_profile": "Crisp lacy edges with mild fermented tang",
        "compounds": ["lactic acid", "amylopectin", "diacetyl"],
        "best_in": ["instant rava dosa", "uttapam", "savory crepes"],
        "avoid_in": ["steamed fluffy idlis"],
        "vegan": False,
    },

    # --- GLOBAL CONDIMENTS, DIPS & SAUCES ---
    {
        "id": "sub_mayonnaise_yogurt_avocado",
        "target": "Mayonnaise (Creamy Sandwich Spread & Dressing Base)",
        "substitute": "Equal parts thick Greek yogurt + 1 tsp olive oil + 1/2 tsp Dijon mustard (or mashed ripe avocado + lime)",
        "tastes": ["fatty_rich", "sour"],
        "cuisine": "Universal Pantry",
        "role": "Thick lipid-protein emulsion that binds salads and prevents soggy bread",
        "flavor_profile": "Creamy, tangy, rich emulsion",
        "compounds": ["lactic acid", "oleic acid", "allyl isothiocyanate"],
        "best_in": ["sandwiches", "coleslaw", "potato salad", "burgers", "wraps"],
        "avoid_in": ["high-heat grilled cheese exterior"],
        "vegan": False,
    },
    {
        "id": "sub_pesto_spinach_walnut",
        "target": "Basil Pesto (Pine Nut & Parmesan Herb Sauce)",
        "substitute": "2 cups baby spinach + 1/2 cup basil/cilantro + 1/3 cup toasted walnuts + 2 tbsp nutritional yeast + olive oil + garlic",
        "tastes": ["fatty_rich", "umami", "cooling", "bitter"],
        "cuisine": "Mediterranean & European",
        "role": "Cold-emulsified herb chlorophyll, nut lipids, and free glutamates",
        "flavor_profile": "Bright herbaceous, nutty, savory garlic-cheese richness",
        "compounds": ["linalool", "estragole", "allicin", "glutamic acid"],
        "best_in": ["pasta", "panini", "roasted vegetables", "pizza base"],
        "avoid_in": ["long-simmered soups (turns brown)"],
        "vegan": True,
    },
    {
        "id": "sub_hummus_white_bean",
        "target": "Chickpea Hummus",
        "substitute": "1 can cannellini (white kidney) beans blended with tahini, lemon juice, garlic, cumin & olive oil",
        "tastes": ["fatty_rich", "sour", "umami"],
        "cuisine": "Middle Eastern & North African",
        "role": "Velvety legume-sesame emulsion with bright citrus-garlic lift",
        "flavor_profile": "Ultra-smooth, creamy, nutty, garlicky citrus",
        "compounds": ["sesamol", "citric acid", "allicin", "cuminaldehyde"],
        "best_in": ["pita wraps", "mezze platters", "veggie dips", "toast"],
        "avoid_in": ["falafel raw batter"],
        "vegan": True,
    },
    {
        "id": "sub_ketchup_tomato_vinegar",
        "target": "Tomato Ketchup",
        "substitute": "2 tbsp tomato paste + 1 tsp apple cider vinegar + 1 tsp maple syrup or jaggery + pinch allspice & salt",
        "tastes": ["sweet", "sour", "umami"],
        "cuisine": "Universal Pantry",
        "role": "Balanced sweet-tart tomato glaze with concentrated glutamates",
        "flavor_profile": "Tangy, sweet, spiced roasted tomato",
        "compounds": ["glutamic acid", "acetic acid", "sucrose", "lycopene"],
        "best_in": ["burgers", "fries", "meatloaf glaze", "bbq sauce"],
        "avoid_in": ["fresh marinara"],
        "vegan": True,
    },
    {
        "id": "sub_cream_cheese_hung_curd",
        "target": "Cream Cheese (Frostings, Cheesecakes & Bagels)",
        "substitute": "1 cup strained hung curd (labneh / thick Greek yogurt) + 1 tbsp softened butter or coconut oil",
        "tastes": ["fatty_rich", "sour"],
        "cuisine": "Baking & Pastry",
        "role": "Dense lactic-acid milkfat paste with spreadable tang",
        "flavor_profile": "Mildly tangy, velvety, rich dairy spread",
        "compounds": ["lactic acid", "diacetyl", "casein"],
        "best_in": ["bagels", "cheesecake", "carrot cake frosting", "dips"],
        "avoid_in": ["high-temp frying"],
        "vegan": False,
    },
    {
        "id": "sub_mozzarella_cashew_cheese",
        "target": "Fresh Mozzarella (Caprese & Pizza)",
        "substitute": "Sliced silken/firm tofu marinated in olive oil, salt & nutritional yeast (or burrata / young provolone)",
        "tastes": ["fatty_rich", "umami", "salty"],
        "cuisine": "Mediterranean & European",
        "role": "Mild, milky, high-moisture protein-lipid contrast to acidic tomatoes",
        "flavor_profile": "Clean, milky, delicate savory softness",
        "compounds": ["casein", "glutamic acid", "oleic acid"],
        "best_in": ["caprese salad", "pizza", "baked pasta", "panini"],
        "avoid_in": ["aged sharp cheddar sauces"],
        "vegan": True,
    },
    {
        "id": "sub_rice_quinoa_cauliflower",
        "target": "1 cup Cooked White Rice (Grain Base)",
        "substitute": "1 cup cooked quinoa, foxtail millet (korra), or riced sauteed cauliflower",
        "tastes": ["sweet", "umami"],
        "cuisine": "Universal Pantry",
        "role": "Neutral starch carrier that absorbs curries, gravies, and stir-fry sauces",
        "flavor_profile": "Mild, fluffy, lightly nutty grain base",
        "compounds": ["amylopectin", "amylose"],
        "best_in": ["curries", "burrito bowls", "fried rice", "dal"],
        "avoid_in": ["sticky sushi rolls"],
        "vegan": True,
    },
    {
        "id": "sub_chicken_soya_jackfruit",
        "target": "250g Boneless Chicken (Curries, Biryani & Stir-Fries)",
        "substitute": "Hydrated soya chunks (meal maker), king oyster mushrooms, or young green jackfruit",
        "tastes": ["umami", "kokumi"],
        "cuisine": "Indian & South Asian",
        "role": "Fibrous, chewy protein matrix that soaks up spice marinades and gravies",
        "flavor_profile": "Meaty, fibrous, savory marinade sponge",
        "compounds": ["soy globulins", "glutamic acid", "1-octen-3-ol"],
        "best_in": ["biryani", "tikka masala", "kebabs", "tacos", "stir-fry"],
        "avoid_in": ["roast whole bird carving"],
        "vegan": True,
    },
]

EXTRA_INGREDIENTS = [
    {"ingredient": "coconut chutney", "tastes": ["fatty_rich", "cooling", "spicy_pungent"], "cuisine": "Indian & South Asian",
     "partners": ["dosa", "idli", "sambar", "curry leaves", "mustard seeds", "green chili", "ginger", "roasted chana dal"],
     "compounds": ["delta-octalactone", "lauric acid", "capsaicin", "linalool"],
     "dishes": ["masala dosa", "idli sambar", "medu vada", "uttapam"]},
    {"ingredient": "peanut chutney", "tastes": ["fatty_rich", "spicy_pungent", "kokumi"], "cuisine": "Indian & South Asian",
     "partners": ["dosa", "idli", "tamarind", "garlic", "red chili", "curry leaves", "mustard seeds"],
     "compounds": ["pyrazines", "oleic acid", "capsaicin", "allicin"],
     "dishes": ["dosa", "idli", "pesarattu", "punugulu"]},
    {"ingredient": "dosa", "tastes": ["sour", "fatty_rich", "kokumi"], "cuisine": "Indian & South Asian",
     "partners": ["coconut chutney", "sambar", "peanut chutney", "kara chutney", "potato masala", "ghee", "gunpowder podi"],
     "compounds": ["lactic acid", "pyrazines", "diacetyl"],
     "dishes": ["masala dosa", "ghee roast dosa", "rava dosa"]},
    {"ingredient": "sambar", "tastes": ["sour", "umami", "spicy_pungent", "kokumi"], "cuisine": "Indian & South Asian",
     "partners": ["idli", "dosa", "rice", "tamarind", "toor dal", "curry leaves", "asafoetida", "drumstick"],
     "compounds": ["tartaric acid", "glutamic acid", "cuminaldehyde", "beta-caryophyllene"],
     "dishes": ["idli sambar", "sambar rice", "medu vada"]},
    {"ingredient": "paneer", "tastes": ["fatty_rich", "sweet", "umami"], "cuisine": "Indian & South Asian",
     "partners": ["tomato", "kasuri methi", "ginger", "garlic", "spinach", "garam masala", "cream", "bell pepper"],
     "compounds": ["casein", "diacetyl", "butyric acid"],
     "dishes": ["paneer tikka", "palak paneer", "butter paneer masala"]},
    {"ingredient": "yogurt", "tastes": ["sour", "fatty_rich", "cooling"], "cuisine": "Universal Pantry",
     "partners": ["cucumber", "mint", "garlic", "cumin", "honey", "berries", "lemon", "chicken"],
     "compounds": ["lactic acid", "diacetyl", "acetaldehyde"],
     "dishes": ["raita", "tzatziki", "tandoori marinade", "smoothies"]},
    {"ingredient": "chickpeas", "tastes": ["umami", "fatty_rich"], "cuisine": "Middle Eastern & North African",
     "partners": ["tahini", "lemon", "garlic", "cumin", "olive oil", "tomato", "spinach", "ginger"],
     "compounds": ["pyrazines", "globulins", "starch"],
     "dishes": ["hummus", "chana masala", "falafel", "roasted chickpeas"]},
    {"ingredient": "egg", "tastes": ["fatty_rich", "umami", "kokumi"], "cuisine": "Universal Pantry",
     "partners": ["butter", "black pepper", "cheese", "chives", "tomato", "bacon", "soy sauce", "rice"],
     "compounds": ["hydrogen sulfide", "lecithin", "glutamic acid"],
     "dishes": ["shakshuka", "fried rice", "omelet", "carbonara"]},
]


def main() -> None:
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 1. Merge & index all substitution rules
    all_subs = {r["id"]: r for r in UNIVERSAL_SUBSTITUTIONS}
    for r in EXTRA_SUBSTITUTIONS:
        all_subs[r["id"]] = r

    try:
        client.delete_collection("flavor_substitutions")
    except Exception:
        pass
    sub_col = client.create_collection("flavor_substitutions", metadata={"hnsw:space": "cosine"})

    sub_ids, sub_docs, sub_metas = [], [], []
    for r in all_subs.values():
        tastes_str = ", ".join(r["tastes"])
        # Natural-language lead sentence improves CrossEncoder calibration significantly
        lead = (
            f"To replace {r['target']} in {', '.join(r['best_in'][:4])}, use {r['substitute']}. "
            f"It delivers {tastes_str} taste via {r['role']}."
        )
        doc = (
            f"{lead}\n"
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
        sub_ids.append(r["id"])
        sub_docs.append(doc)
        sub_metas.append(
            {
                "target": r["target"],
                "substitute": r["substitute"],
                "tastes": tastes_str,
                "cuisine": r["cuisine"],
                "vegan": bool(r["vegan"]),
            }
        )

    sub_vecs = model.encode(sub_docs, batch_size=64, normalize_embeddings=True).tolist()
    sub_col.add(ids=sub_ids, documents=sub_docs, embeddings=sub_vecs, metadatas=sub_metas)

    # 2. Merge & index all ingredient taste profiles
    ing_map = {item["ingredient"].lower(): dict(item) for item in UNIVERSAL_INGREDIENTS}
    for item in EXTRA_INGREDIENTS:
        ing_map[item["ingredient"].lower()] = dict(item)
    merge_local_recipes(RAW_DIR / "recipes_25k.jsonl", ing_map, max_extra=500)

    try:
        client.delete_collection("flavor_pairings")
    except Exception:
        pass
    pair_col = client.create_collection("flavor_pairings", metadata={"hnsw:space": "cosine"})

    pair_ids, pair_docs, pair_metas = [], [], []
    for idx, (ing, data) in enumerate(ing_map.items()):
        tastes_str = ", ".join(data["tastes"])
        lead = (
            f"{ing.title()} brings {tastes_str} taste in {data['cuisine']} cooking and pairs well with "
            f"{', '.join(data['partners'][:6])} in dishes like {', '.join(data['dishes'][:3])}."
        )
        lines = [
            lead,
            f"Ingredient: {ing}",
            f"Taste Dimensions: {tastes_str}",
            f"Culinary Corner: {data['cuisine']}",
            f"Top Culinary Pairing Partners: {', '.join(data['partners'])}",
            f"Best Dish Contexts: {', '.join(data['dishes'])}",
        ]
        if data["compounds"]:
            lines.append(f"Key Volatile & Flavor Compounds: {', '.join(data['compounds'])}")

        pair_ids.append(f"ing_{idx}_{re.sub(r'[^a-z0-9]+', '_', ing)[:40]}")
        pair_docs.append("\n".join(lines))
        pair_metas.append(
            {
                "ingredient": ing,
                "tastes": tastes_str,
                "cuisine": data["cuisine"],
                "partners": ", ".join(data["partners"][:8]),
                "compounds": ", ".join(data["compounds"][:6]),
            }
        )

    pair_vecs = model.encode(pair_docs, batch_size=64, normalize_embeddings=True).tolist()
    pair_col.add(ids=pair_ids, documents=pair_docs, embeddings=pair_vecs, metadatas=pair_metas)

    print(f"Indexed {sub_col.count()} global substitution rules.")
    print(f"Indexed {pair_col.count()} global ingredient & dish taste profiles.")


if __name__ == "__main__":
    main()
