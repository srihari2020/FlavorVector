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

# ==============================================================================
# 1. UNIVERSAL SUBSTITUTIONS (EVERY TASTE + EVERY CULINARY CORNER)
# ==============================================================================
UNIVERSAL_SUBSTITUTIONS = [
    # --- SWEET (Sugars, Syrups, Fruits, Mithai & Baking) ---
    {
        "id": "sub_sugar_date_paste",
        "target": "White Granulated Sugar (Shakes, Smoothies & Energy Bites)",
        "substitute": "2 to 3 pitted Medjool dates (or 2 tbsp date paste) per 2 tbsp sugar",
        "tastes": ["sweet", "fatty_rich"],
        "cuisine": "Universal Pantry",
        "role": "Fructose-glucose sweetening plus soluble pectin fiber that thickens cold drinks",
        "flavor_profile": "Caramel, honeyed sweetness, velvety body",
        "compounds": ["fructose", "glucose", "pectin", "polyphenols"],
        "best_in": ["banana shakes", "smoothies", "milkshakes", "oatmeal", "energy balls"],
        "avoid_in": ["clear syrups", "airy sponge cakes"],
        "vegan": True,
    },
    {
        "id": "sub_sugar_jaggery",
        "target": "Brown Sugar or White Sugar (Caramel & Mineral Depth)",
        "substitute": "1 cup grated jaggery (gur) or palm/coconut sugar per 1 cup sugar",
        "tastes": ["sweet", "kokumi"],
        "cuisine": "Indian & South Asian",
        "role": "Unrefined sucrose with trace minerals and molasses for deep caramelization",
        "flavor_profile": "Earthy molasses, toffee, warm roasted cane",
        "compounds": ["sucrose", "pyrazines", "furans", "maltol"],
        "best_in": ["banana shakes", "chai", "kheer", "payasam", "cookies", "bbq sauce"],
        "avoid_in": ["snow-white frostings", "meringues"],
        "vegan": True,
    },
    {
        "id": "sub_sugar_maple",
        "target": "White Granulated Sugar (Cold Liquids & Glazes)",
        "substitute": "3/4 cup pure maple syrup per 1 cup sugar (reduce liquid by 3 tbsp)",
        "tastes": ["sweet"],
        "cuisine": "Universal Pantry",
        "role": "Dissolves instantly without grittiness; adds hygroscopic moisture",
        "flavor_profile": "Clean sweetness with subtle woody vanillin notes",
        "compounds": ["sucrose", "vanillin", "syringaldehyde", "malic acid"],
        "best_in": ["banana shakes", "iced coffee", "pancakes", "glazes", "vinaigrettes"],
        "avoid_in": ["crisp shortbread cookies"],
        "vegan": True,
    },
    {
        "id": "sub_sugar_honey",
        "target": "White Granulated Sugar (Floral Sweetness)",
        "substitute": "3/4 cup raw honey per 1 cup sugar",
        "tastes": ["sweet"],
        "cuisine": "Universal Pantry",
        "role": "Invert-sugar sweetener that retains moisture and browns faster in baking",
        "flavor_profile": "Floral, warm fructose sweetness",
        "compounds": ["fructose", "glucose", "phenylacetic acid", "linalool"],
        "best_in": ["shakes", "tea", "yogurt bowls", "dressings", "marinades"],
        "avoid_in": ["strictly vegan recipes"],
        "vegan": False,
    },
    {
        "id": "sub_sugar_monkfruit_stevia",
        "target": "White Sugar (Zero-Calorie Sweetness)",
        "substitute": "1:1 monk fruit erythritol blend or 1 tsp liquid stevia per 1 cup sugar",
        "tastes": ["sweet"],
        "cuisine": "Universal Pantry",
        "role": "High-intensity glycoside sweetness with zero glycemic impact",
        "flavor_profile": "Clean sweet finish without bulk or browning",
        "compounds": ["mogroside V", "rebaudioside A"],
        "best_in": ["protein shakes", "banana shakes", "iced tea", "coffee"],
        "avoid_in": ["caramel", "yeast fermentation"],
        "vegan": True,
    },
    {
        "id": "sub_condensed_milk_coconut",
        "target": "Sweetened Condensed Milk (Mithai, Fudge & Iced Coffee)",
        "substitute": "1 can full-fat coconut milk simmered with 1/2 cup sugar or jaggery until reduced by half",
        "tastes": ["sweet", "fatty_rich"],
        "cuisine": "Indian & South Asian",
        "role": "Thick caramelized milk-fat and sugar emulsion for dense desserts",
        "flavor_profile": "Rich creamy sweetness with subtle coconut notes",
        "compounds": ["sucrose", "lauric acid", "delta-octalactone"],
        "best_in": ["barfi", "fudge", "Vietnamese iced coffee", "tres leches", "ice cream"],
        "avoid_in": ["neutral dairy-pure custards"],
        "vegan": True,
    },

    # --- SOUR / ACIDIC (Citrus, Vinegars, Tamarind, Amchur, Sumac) ---
    {
        "id": "sub_buttermilk_lemon",
        "target": "Buttermilk (Tangy Dairy Leavener)",
        "substitute": "1 cup whole milk + 1 tbsp fresh lemon juice (rest 10 min)",
        "tastes": ["sour", "fatty_rich"],
        "cuisine": "Baking & Pastry",
        "role": "Acidic liquid that tenderizes gluten and activates baking soda for lift",
        "flavor_profile": "Tangy citrus brightness with mild dairy creaminess",
        "compounds": ["citric acid", "lactic acid", "milk triglycerides"],
        "best_in": ["pancakes", "waffles", "biscuits", "quick breads", "fried chicken marinade"],
        "avoid_in": ["thick dressings without emulsifier"],
        "vegan": False,
    },
    {
        "id": "sub_buttermilk_yogurt",
        "target": "Buttermilk (Cultured Thickness)",
        "substitute": "3/4 cup plain dahi/yogurt + 1/4 cup water",
        "tastes": ["sour", "fatty_rich"],
        "cuisine": "Universal Pantry",
        "role": "Lactic acid for baking soda reaction plus cultured viscosity",
        "flavor_profile": "Rich cultured tang, creamy dairy finish",
        "compounds": ["lactic acid", "diacetyl", "acetaldehyde"],
        "best_in": ["pancakes", "muffins", "cakes", "kadhi", "marinades"],
        "avoid_in": ["thin crepe batters"],
        "vegan": False,
    },
    {
        "id": "sub_buttermilk_vegan",
        "target": "Buttermilk (Plant-Based)",
        "substitute": "1 cup unsweetened soy milk + 1 tbsp apple cider vinegar",
        "tastes": ["sour"],
        "cuisine": "Baking & Pastry",
        "role": "Curdles soy proteins for body and supplies acetic acid for baking soda",
        "flavor_profile": "Sharp tang, neutral-nutty base",
        "compounds": ["acetic acid", "soy proteins"],
        "best_in": ["vegan pancakes", "cornbread", "quick breads", "vegan cakes"],
        "avoid_in": ["egg custards"],
        "vegan": True,
    },
    {
        "id": "sub_tamarind_amchur_lime",
        "target": "1 tbsp Tamarind Paste (Fruity Tartness)",
        "substitute": "1 tsp amchur (dried mango powder) + 1 tsp lime juice + 1/2 tsp jaggery",
        "tastes": ["sour", "sweet", "astringent"],
        "cuisine": "Indian & South Asian",
        "role": "Tartaric/citric fruit acidity that cuts heavy spices and lentils",
        "flavor_profile": "Deep, fruity, tangy-sweet sourness",
        "compounds": ["tartaric acid", "citric acid", "malic acid", "furfural"],
        "best_in": ["sambar", "rasam", "chutneys", "pad thai", "chaat"],
        "avoid_in": ["clear European broths"],
        "vegan": True,
    },
    {
        "id": "sub_kokum_tamarind_lime",
        "target": "Kokum Petals (Coastal Souring & Cooling)",
        "substitute": "1 tbsp tamarind pulp + splash of pomegranate juice or lime juice",
        "tastes": ["sour", "astringent", "cooling"],
        "cuisine": "Indian & South Asian",
        "role": "Hydroxycitric acidity that brightens coconut curries without turning them murky",
        "flavor_profile": "Ruby-tart, fruity, clean coastal sourness",
        "compounds": ["hydroxycitric acid", "garcinol", "anthocyanins"],
        "best_in": ["sol kadhi", "fish curry", "dal", "coastal stews"],
        "avoid_in": ["cream-based sauces (can curdle)"],
        "vegan": True,
    },
    {
        "id": "sub_sumac_lemon_zest",
        "target": "1 tbsp Ground Sumac (Dry Citrus Tang)",
        "substitute": "1 tsp finely grated lemon zest + 1/2 tsp citric acid or splash of lemon juice",
        "tastes": ["sour", "astringent"],
        "cuisine": "Middle Eastern & North African",
        "role": "Dry malic-acid seasoning that adds bright sourness without extra liquid",
        "flavor_profile": "Floral, tart, astringent berry-citrus tang",
        "compounds": ["malic acid", "tannins", "limonene"],
        "best_in": ["fattoush salad", "kebabs", "hummus", "roasted meats", "rice"],
        "avoid_in": ["sweet pastries"],
        "vegan": True,
    },
    {
        "id": "sub_rice_vinegar_acv",
        "target": "1 tbsp Rice Vinegar",
        "substitute": "1 tbsp apple cider vinegar + 1/4 tsp sugar",
        "tastes": ["sour", "sweet"],
        "cuisine": "East & Southeast Asian",
        "role": "Gentle ~4.5% acetic acidity without harsh bite",
        "flavor_profile": "Mild, slightly sweet fruit acidity",
        "compounds": ["acetic acid", "malic acid"],
        "best_in": ["sushi rice", "stir-fries", "pickled cucumbers", "dumpling sauce"],
        "avoid_in": ["crystal-clear glazes"],
        "vegan": True,
    },
    {
        "id": "sub_white_wine_broth_vinegar",
        "target": "1/2 cup Dry White Wine (Deglazing & Pan Sauces)",
        "substitute": "1/2 cup vegetable or chicken broth + 1 tbsp white wine vinegar or lemon juice",
        "tastes": ["sour", "umami"],
        "cuisine": "Mediterranean & European",
        "role": "Acidity lifts caramelized fond from pans and balances butter/cream",
        "flavor_profile": "Crisp, bright acidity with savory depth",
        "compounds": ["acetic acid", "citric acid", "glutamic acid"],
        "best_in": ["risotto", "chicken piccata", "pan sauces", "mussels"],
        "avoid_in": ["wine-poached pears"],
        "vegan": True,
    },

    # --- UMAMI & SALTY (Glutamates, Ribonucleotides, Ferments) ---
    {
        "id": "sub_fish_sauce_umami",
        "target": "1 tbsp Fish Sauce",
        "substitute": "1 tbsp soy sauce + 1/4 tsp rice vinegar + pinch of crushed nori or mushroom powder",
        "tastes": ["umami", "salty", "sour"],
        "cuisine": "East & Southeast Asian",
        "role": "Glutamate + guanylate/inosinate umami synergy and salinity",
        "flavor_profile": "Deep savory oceanic umami with balanced salt-acid",
        "compounds": ["glutamic acid", "inosinate", "guanylate", "sodium chloride"],
        "best_in": ["curries", "stir-fries", "broths", "dipping sauces", "fried rice"],
        "avoid_in": ["raw dishes requiring anchovy funk"],
        "vegan": True,
    },
    {
        "id": "sub_oyster_sauce_mushroom",
        "target": "1 tbsp Oyster Sauce",
        "substitute": "1 tbsp vegetarian mushroom stir-fry sauce (or 2 tsp soy sauce + 1 tsp hoisin + pinch sugar)",
        "tastes": ["umami", "sweet", "salty", "kokumi"],
        "cuisine": "East & Southeast Asian",
        "role": "Glossy caramelized umami glaze that clings to vegetables and noodles",
        "flavor_profile": "Thick, savory-sweet roasted umami",
        "compounds": ["glutamic acid", "guanylate", "sucrose", "furans"],
        "best_in": ["chow mein", "blanched greens", "beef & broccoli", "fried rice"],
        "avoid_in": ["light clear soups"],
        "vegan": True,
    },
    {
        "id": "sub_miso_tahini_soy",
        "target": "1 tbsp White or Red Miso Paste",
        "substitute": "2 tsp tahini + 1 tsp soy sauce + pinch of nutritional yeast",
        "tastes": ["umami", "salty", "fatty_rich", "kokumi"],
        "cuisine": "East & Southeast Asian",
        "role": "Fermented paste body, salinity, and free amino-acid depth",
        "flavor_profile": "Nutty, salty, fermented savory richness",
        "compounds": ["glutamic acid", "methional", "sesamol"],
        "best_in": ["ramen broth", "salad dressings", "glazes", "marinades"],
        "avoid_in": ["traditional miso soup"],
        "vegan": True,
    },
    {
        "id": "sub_parmesan_nooch",
        "target": "1/4 cup Grated Parmesan Cheese",
        "substitute": "3 tbsp nutritional yeast + 1 tbsp finely ground raw cashews + 1/4 tsp salt",
        "tastes": ["umami", "salty", "fatty_rich"],
        "cuisine": "Mediterranean & European",
        "role": "Free glutamates, guanylates, and nutty lipids replicate aged cheese",
        "flavor_profile": "Savory, nutty, aged-cheesy umami",
        "compounds": ["glutamic acid", "guanylate", "oleic acid"],
        "best_in": ["pasta", "pesto", "roasted vegetables", "risotto", "popcorn"],
        "avoid_in": ["melted cheese crisps"],
        "vegan": True,
    },
    {
        "id": "sub_anchovy_capers_miso",
        "target": "2 Anchovy Fillets (Savory Base)",
        "substitute": "1 tbsp chopped capers + 1/2 tsp miso paste + pinch of kelp/nori",
        "tastes": ["salty", "umami", "sour"],
        "cuisine": "Mediterranean & European",
        "role": "Briny salinity and glutamate depth that melts into pasta sauces and dressings",
        "flavor_profile": "Briny, punchy, savory-marine umami",
        "compounds": ["glutamic acid", "inosinate", "rutin", "sodium chloride"],
        "best_in": ["caesar dressing", "puttanesca", "bagna cauda", "roasted lamb"],
        "avoid_in": ["whole-fillet garnishes"],
        "vegan": True,
    },
    {
        "id": "sub_tomato_paste_sundried",
        "target": "1 tbsp Tomato Paste",
        "substitute": "2 tbsp pureed sun-dried tomatoes or 3 tbsp reduced tomato passata",
        "tastes": ["umami", "sour", "sweet"],
        "cuisine": "Universal Pantry",
        "role": "Concentrated lycopene, natural glutamates, and tart-sweet body",
        "flavor_profile": "Rich roasted tomato umami and acidity",
        "compounds": ["glutamic acid", "lycopene", "citric acid"],
        "best_in": ["curries", "pasta sauces", "stews", "braises"],
        "avoid_in": ["clear broths"],
        "vegan": True,
    },

    # --- SPICY / PUNGENT (Chilies, Mustard, Ginger, Garlic, Peppercorns) ---
    {
        "id": "sub_asafoetida_garlic_onion",
        "target": "1/4 tsp Asafoetida (Hing) in Tadka",
        "substitute": "1/2 tsp garlic powder + 1/2 tsp onion powder bloomed in hot oil/ghee",
        "tastes": ["spicy_pungent", "umami", "kokumi"],
        "cuisine": "Indian & South Asian",
        "role": "Sulfurous allium aroma base when tempered in hot fat for lentils",
        "flavor_profile": "Savory, pungent roasted leek-garlic warmth",
        "compounds": ["disulfides", "ferulic acid", "alliin"],
        "best_in": ["dal tadka", "sambar", "aloo jeera", "kadhi"],
        "avoid_in": ["strict Jain/no-allium diets"],
        "vegan": True,
    },
    {
        "id": "sub_gochujang_miso_chili",
        "target": "1 tbsp Gochujang (Korean Chili Paste)",
        "substitute": "1 tbsp red miso + 1 tsp gochugaru or cayenne + 1 tsp honey/maple + 1/2 tsp rice vinegar",
        "tastes": ["spicy_pungent", "umami", "sweet", "salty"],
        "cuisine": "East & Southeast Asian",
        "role": "Fermented glutinous-rice chili paste with heat, sweetness, and body",
        "flavor_profile": "Smoky-sweet, fermented savory chili heat",
        "compounds": ["capsaicin", "glutamic acid", "maltose"],
        "best_in": ["bibimbap", "spicy stir-fries", "marinades", "wings glaze"],
        "avoid_in": ["clear soups"],
        "vegan": True,
    },
    {
        "id": "sub_chipotle_smoked_paprika",
        "target": "1 Chipotle Pepper in Adobo",
        "substitute": "1 tsp smoked paprika + 1/4 tsp cayenne + 1 tsp tomato paste + 1/2 tsp cider vinegar",
        "tastes": ["spicy_pungent", "umami", "sour"],
        "cuisine": "Latin American",
        "role": "Wood-smoked capsaicin heat with tangy adobo depth",
        "flavor_profile": "Smoky, earthy, medium-hot chili warmth",
        "compounds": ["capsaicin", "guaiacol", "syringol", "capsanthin"],
        "best_in": ["tacos", "chili con carne", "salsa", "chipotle mayo", "marinades"],
        "avoid_in": ["delicate white fish"],
        "vegan": True,
    },
    {
        "id": "sub_harissa_spices",
        "target": "1 tbsp Harissa Paste",
        "substitute": "1 tsp smoked paprika + 1/2 tsp cayenne + 1/2 tsp ground cumin + 1/2 tsp coriander + 1 minced garlic clove + olive oil",
        "tastes": ["spicy_pungent", "bitter"],
        "cuisine": "Middle Eastern & North African",
        "role": "Aromatic roasted chili-spice paste with earthy warmth",
        "flavor_profile": "Warm, smoky, garlicky North African chili spice",
        "compounds": ["capsaicin", "cuminaldehyde", "linalool", "allicin"],
        "best_in": ["couscous", "roasted carrots", "shakshuka", "grilled lamb", "chickpeas"],
        "avoid_in": ["East Asian stir-fries"],
        "vegan": True,
    },
    {
        "id": "sub_fresh_ginger_galangal",
        "target": "1 tbsp Fresh Galangal",
        "substitute": "1 tbsp fresh ginger + 1/4 tsp ground black pepper + pinch of lemon zest",
        "tastes": ["spicy_pungent", "cooling"],
        "cuisine": "East & Southeast Asian",
        "role": "Piney, peppery rhizome heat for Southeast Asian curry pastes",
        "flavor_profile": "Sharp, citrusy-piney pepper warmth",
        "compounds": ["acetoxychavicol acetate", "1,8-cineole", "gingerol"],
        "best_in": ["tom kha gai", "Thai curry paste", "laksa", "rendang"],
        "avoid_in": ["gingerbread cookies"],
        "vegan": True,
    },
    {
        "id": "sub_wasabi_horseradish_mustard",
        "target": "1 tsp Real Wasabi",
        "substitute": "1 tsp prepared horseradish + 1/4 tsp dry English mustard powder",
        "tastes": ["spicy_pungent"],
        "cuisine": "East & Southeast Asian",
        "role": "Volatile nasal-sinus heat from isothiocyanates (clears quickly unlike chili)",
        "flavor_profile": "Sharp, pungent, vegetal sinus heat",
        "compounds": ["allyl isothiocyanate", "sinigrin"],
        "best_in": ["sushi", "soba noodles", "roast beef", "mayo dips"],
        "avoid_in": ["long-simmered stews (heat evaporates)"],
        "vegan": True,
    },

    # --- BITTER & ASTRINGENT (Cocoa, Coffee, Fenugreek, Tea, Greens) ---
    {
        "id": "sub_kasuri_methi_celery_fennel",
        "target": "1 tbsp Kasuri Methi (Dried Fenugreek Leaves)",
        "substitute": "1 tbsp dried celery leaves + 1/4 tsp crushed fennel seeds",
        "tastes": ["bitter", "sweet"],
        "cuisine": "Indian & South Asian",
        "role": "Sotolon-driven maple-bitter aroma that cuts heavy cream/tomato gravies",
        "flavor_profile": "Bittersweet, hay-like, maple-herbaceous warmth",
        "compounds": ["sotolon", "anethole", "phthalides"],
        "best_in": ["butter chicken", "paneer tikka masala", "dal makhani", "aloo methi"],
        "avoid_in": ["sweet desserts"],
        "vegan": True,
    },
    {
        "id": "sub_cocoa_dark_chocolate",
        "target": "3 tbsp Unsweetened Cocoa Powder",
        "substitute": "1 oz (28g) unsweetened dark baking chocolate (reduce recipe fat/butter by 1 tbsp)",
        "tastes": ["bitter", "fatty_rich"],
        "cuisine": "Baking & Pastry",
        "role": "Roasted cacao solids provide bitterness, color, and structure",
        "flavor_profile": "Deep roasted, bitter-earthy chocolate",
        "compounds": ["theobromine", "pyrazines", "epicatechin"],
        "best_in": ["brownies", "chocolate cake", "mole sauce", "hot chocolate"],
        "avoid_in": ["dry spice rubs"],
        "vegan": True,
    },
    {
        "id": "sub_espresso_dark_cocoa_chicory",
        "target": "1 tsp Espresso Powder (Flavor Enhancer)",
        "substitute": "1 tsp roasted chicory powder or 1 tbsp strong brewed dark roast coffee",
        "tastes": ["bitter", "kokumi"],
        "cuisine": "Baking & Pastry",
        "role": "Roasted Maillard bitterness deepens chocolate and savory beef gravies",
        "flavor_profile": "Toasted, bitter, caramelized coffee notes",
        "compounds": ["chlorogenic acid lactones", "2-furfurylthiol", "pyrazines"],
        "best_in": ["chocolate cake", "brownies", "beef stew", "steak rub"],
        "avoid_in": ["light citrus desserts"],
        "vegan": True,
    },
    {
        "id": "sub_pomegranate_molasses_balsamic",
        "target": "1 tbsp Pomegranate Molasses",
        "substitute": "1 tbsp aged balsamic vinegar + 1 tsp lemon juice + 1/2 tsp honey/maple",
        "tastes": ["sour", "astringent", "sweet"],
        "cuisine": "Middle Eastern & North African",
        "role": "Tannic, tart-sweet glaze that balances grilled meats and walnuts",
        "flavor_profile": "Deep ruby fruit tartness, tannic astringency, sweet finish",
        "compounds": ["punicalagins", "citric acid", "malic acid", "tannins"],
        "best_in": ["muhammara", "fesenjan", "fattoush", "roasted eggplant"],
        "avoid_in": ["vanilla custards"],
        "vegan": True,
    },

    # --- COOLING & AROMATIC (Mint, Cardamom, Star Anise, Curry Leaves) ---
    {
        "id": "sub_curry_leaves_bay_lime",
        "target": "10 Fresh Curry Leaves (Kadi Patta)",
        "substitute": "1 bay leaf + 1/2 tsp finely grated lime zest bloomed in hot oil",
        "tastes": ["cooling", "bitter"],
        "cuisine": "Indian & South Asian",
        "role": "Citrus-herbal terpenes released during oil tempering (tadka)",
        "flavor_profile": "Bright, nutty, citrus-pine aromatic freshness",
        "compounds": ["beta-caryophyllene", "linalool", "limonene", "pinene"],
        "best_in": ["sambar", "poha", "upma", "coconut chutney", "rasam"],
        "avoid_in": ["raw salads"],
        "vegan": True,
    },
    {
        "id": "sub_cardamom_cinnamon_nutmeg",
        "target": "1 tsp Ground Green Cardamom (Elaichi)",
        "substitute": "1/2 tsp ground cinnamon + 1/4 tsp nutmeg + pinch of ground clove",
        "tastes": ["cooling", "sweet"],
        "cuisine": "Indian & South Asian",
        "role": "Eucalyptol-terpinyl floral lift in both sweet mithai and savory biryanis",
        "flavor_profile": "Floral, eucalyptus-citrus, warm sweet spice",
        "compounds": ["1,8-cineole", "alpha-terpinyl acetate", "linalool"],
        "best_in": ["masala chai", "biryani", "kheer", "cinnamon buns", "coffee"],
        "avoid_in": ["delicate seafood"],
        "vegan": True,
    },
    {
        "id": "sub_fresh_mint_basil_oregano",
        "target": "1/4 cup Fresh Mint Leaves",
        "substitute": "3 tbsp sweet basil + 1 tbsp flat-leaf parsley (or tiny pinch of fennel)",
        "tastes": ["cooling", "sweet"],
        "cuisine": "Universal Pantry",
        "role": "Herbaceous, bright aromatic lift that cuts rich lamb, yogurt, or fruit",
        "flavor_profile": "Fresh, sweet-herbal, cooling aroma",
        "compounds": ["menthol", "carvone", "linalool", "estragole"],
        "best_in": ["raita", "chutneys", "tabbouleh", "lamb dishes", "summer salads"],
        "avoid_in": ["peppermint candy"],
        "vegan": True,
    },

    # --- FATTY / RICH & KOKUMI (Dairy, Oils, Nuts, Broths) ---
    {
        "id": "sub_paneer_tofu_halloumi",
        "target": "200g Paneer (Indian Cottage Cheese)",
        "substitute": "200g pressed extra-firm tofu (tossed with 1/2 tsp lemon juice & pinch salt) or halloumi",
        "tastes": ["fatty_rich", "umami"],
        "cuisine": "Indian & South Asian",
        "role": "Non-melting protein-fat cubes that absorb spiced gravies without disintegrating",
        "flavor_profile": "Milky, mild, spongy-firm texture",
        "compounds": ["soy globulins", "casein", "milk triglycerides"],
        "best_in": ["palak paneer", "matar paneer", "tikka skewers", "stir-fries"],
        "avoid_in": ["melty grilled cheese"],
        "vegan": True,
    },
    {
        "id": "sub_ghee_browned_butter",
        "target": "1 tbsp Ghee (Clarified Butter)",
        "substitute": "1 tbsp browned unsalted butter (or 3/4 tbsp refined coconut oil + pinch turmeric for vegan)",
        "tastes": ["fatty_rich", "kokumi"],
        "cuisine": "Indian & South Asian",
        "role": "High-smoke-point milkfat carrying fat-soluble spice aromatics",
        "flavor_profile": "Nutty, caramelized butterscotch-dairy richness",
        "compounds": ["butyric acid", "delta-decalactone", "pyrazines"],
        "best_in": ["biryani", "dal tadka", "halwa", "roasting vegetables"],
        "avoid_in": ["vegan dishes unless using coconut oil"],
        "vegan": False,
    },
    {
        "id": "sub_heavy_cream_cashew",
        "target": "1 cup Heavy Cream (Savory Gravies & Pasta)",
        "substitute": "1/2 cup soaked raw cashews blended silky-smooth with 3/4 cup water",
        "tastes": ["fatty_rich", "sweet"],
        "cuisine": "Universal Pantry",
        "role": "Stable plant lipid-starch emulsion that resists curdling in acidic tomato gravies",
        "flavor_profile": "Velvety, mildly sweet, rich cream mouthfeel",
        "compounds": ["oleic acid", "linoleic acid", "amylopectin"],
        "best_in": ["butter masala", "korma", "alfredo pasta", "creamy tomato soup"],
        "avoid_in": ["whipped cream peaks"],
        "vegan": True,
    },
    {
        "id": "sub_heavy_cream_coconut",
        "target": "1 cup Heavy Whipping Cream (Desserts & Curries)",
        "substitute": "1 cup chilled full-fat coconut cream (thick top layer from canned coconut milk)",
        "tastes": ["fatty_rich", "sweet"],
        "cuisine": "East & Southeast Asian",
        "role": "~35% saturated fat emulsion capable of whipping or enriching curries",
        "flavor_profile": "Lush, velvety richness with tropical coconut notes",
        "compounds": ["lauric acid", "delta-octalactone", "gamma-nonalactone"],
        "best_in": ["Thai curries", "ganache", "whipped topping", "shakes", "ice cream"],
        "avoid_in": ["classic French cream sauces"],
        "vegan": True,
    },
    {
        "id": "sub_tahini_sunflower_cashew",
        "target": "1/4 cup Tahini (Sesame Paste)",
        "substitute": "1/4 cup unsweetened sunflower seed butter or cashew butter + 1/2 tsp toasted sesame oil",
        "tastes": ["fatty_rich", "bitter", "kokumi"],
        "cuisine": "Middle Eastern & North African",
        "role": "Emulsifying roasted seed lipids that bind lemon and garlic in hummus",
        "flavor_profile": "Toasted, slightly bitter, nutty richness",
        "compounds": ["sesamol", "pyrazines", "oleic acid"],
        "best_in": ["hummus", "baba ganoush", "tahini dressing", "halva"],
        "avoid_in": ["nut-free schools if using cashew"],
        "vegan": True,
    },
    {
        "id": "sub_beef_broth_mushroom_kokumi",
        "target": "1 cup Beef Broth (Deep Kokumi & Meatiness)",
        "substitute": "1 cup dried shiitake mushroom broth + 1 tsp soy sauce + 1/2 tsp tomato paste",
        "tastes": ["umami", "kokumi", "salty"],
        "cuisine": "Universal Pantry",
        "role": "Guanylates + glutamates + roasted Maillard notes mimic slow-simmered collagen",
        "flavor_profile": "Deep, lingering, mouth-coating savory richness",
        "compounds": ["guanylate", "glutamic acid", "1-octen-3-ol", "methional"],
        "best_in": ["French onion soup", "risotto", "gravy", "stews", "ramen"],
        "avoid_in": ["light fruit glazes"],
        "vegan": True,
    },

    # --- STRUCTURAL BAKING & BINDERS (Eggs, Flours, Leaveners, Thickeners) ---
    {
        "id": "sub_egg_leavener",
        "target": "1 Large Egg (Leavening & Airy Lift)",
        "substitute": "1 tsp baking soda + 1 tbsp white or apple cider vinegar",
        "tastes": ["sour"],
        "cuisine": "Baking & Pastry",
        "role": "Rapid CO2 gas release for light, airy cake crumb",
        "flavor_profile": "Neutral",
        "compounds": ["sodium bicarbonate", "acetic acid"],
        "best_in": ["cupcakes", "sponge cakes", "quick breads"],
        "avoid_in": ["meatballs", "custards", "dense brownies"],
        "vegan": True,
    },
    {
        "id": "sub_egg_binder",
        "target": "1 Large Egg (Binding & Moisture)",
        "substitute": "1 tbsp ground flaxseed + 3 tbsp warm water (rest 10 min until gelled)",
        "tastes": ["fatty_rich"],
        "cuisine": "Baking & Pastry",
        "role": "Polysaccharide mucilage gel binds particles without cholesterol",
        "flavor_profile": "Mild, earthy, toasted nutty",
        "compounds": ["flax mucilage", "alpha-linolenic acid"],
        "best_in": ["meatballs", "veggie burgers", "oatmeal cookies", "pancakes"],
        "avoid_in": ["meringues", "angel food cake"],
        "vegan": True,
    },
    {
        "id": "sub_egg_aquafaba",
        "target": "1 Egg White (Whipping Foam & Emulsion)",
        "substitute": "3 tbsp aquafaba (chickpea cooking liquid) whipped with pinch cream of tartar",
        "tastes": ["umami"],
        "cuisine": "Baking & Pastry",
        "role": "Plant saponins and albumins trap air bubbles into glossy meringue peaks",
        "flavor_profile": "Neutral once whipped with sugar or acid",
        "compounds": ["saponins", "globulins", "albumins"],
        "best_in": ["meringues", "chocolate mousse", "vegan mayo", "macarons"],
        "avoid_in": ["omelets"],
        "vegan": True,
    },
    {
        "id": "sub_besan_chickpea_flour",
        "target": "1 cup Besan (Gram / Chickpea Flour)",
        "substitute": "3/4 cup yellow mung bean flour (moong dal flour) or 1/2 cup cornmeal + 1/2 cup AP flour",
        "tastes": ["umami", "fatty_rich"],
        "cuisine": "Indian & South Asian",
        "role": "High-protein legume flour that crisps in oil and binds batters without gluten",
        "flavor_profile": "Nutty, earthy, roasted legume",
        "compounds": ["globulins", "pyrazines", "amylose"],
        "best_in": ["pakoras", "chilla", "kadhi", "dhokla", "socca"],
        "avoid_in": ["yeast sandwich bread"],
        "vegan": True,
    },
    {
        "id": "sub_cornstarch_arrowroot",
        "target": "1 tbsp Cornstarch (Glossy Thickener)",
        "substitute": "1 tbsp arrowroot powder or potato starch (mixed in cold water slurry)",
        "tastes": ["sweet"],
        "cuisine": "Universal Pantry",
        "role": "Rapid starch gelatinization at lower heat with crystal-clear sheen",
        "flavor_profile": "Neutral, silky glossy finish",
        "compounds": ["amylopectin", "amylose"],
        "best_in": ["stir-fry sauces", "fruit glazes", "gravies", "soups"],
        "avoid_in": ["dairy puddings (turns slimy)"],
        "vegan": True,
    },
]

# ==============================================================================
# 2. UNIVERSAL INGREDIENT & TASTE GRAPH (EVERY TASTE + EVERY CUISINE CORNER)
# ==============================================================================
UNIVERSAL_INGREDIENTS = [
    # --- INDIAN & SOUTH ASIAN ---
    {"ingredient": "cumin", "tastes": ["spicy_pungent", "bitter"], "cuisine": "Indian & South Asian",
     "partners": ["coriander", "turmeric", "garlic", "ginger", "chili", "lemon", "chickpeas", "potatoes"],
     "compounds": ["cuminaldehyde", "gamma-terpinene", "beta-pinene"],
     "dishes": ["jeera rice", "dal tadka", "tacos", "roasted cauliflower"]},
    {"ingredient": "coriander", "tastes": ["sweet", "cooling"], "cuisine": "Indian & South Asian",
     "partners": ["cumin", "turmeric", "garlic", "ginger", "lime", "chili", "cilantro", "chicken"],
     "compounds": ["linalool", "alpha-pinene", "geranyl acetate"],
     "dishes": ["curry", "spiced rubs", "falafel", "sambar"]},
    {"ingredient": "turmeric", "tastes": ["bitter", "spicy_pungent"], "cuisine": "Indian & South Asian",
     "partners": ["cumin", "coriander", "ginger", "garlic", "black pepper", "coconut milk", "cauliflower"],
     "compounds": ["curcumin", "ar-turmerone", "zingiberene"],
     "dishes": ["golden curry", "dal", "roasted cauliflower", "golden milk"]},
    {"ingredient": "cardamom", "tastes": ["cooling", "sweet"], "cuisine": "Indian & South Asian",
     "partners": ["cinnamon", "ginger", "clove", "vanilla", "orange", "espresso", "rice", "pistachio"],
     "compounds": ["1,8-cineole", "alpha-terpinyl acetate", "linalool"],
     "dishes": ["masala chai", "biryani", "kheer", "spiced buns"]},
    {"ingredient": "tamarind", "tastes": ["sour", "sweet", "astringent"], "cuisine": "Indian & South Asian",
     "partners": ["jaggery", "cumin", "chili", "curry leaves", "coconut", "fish sauce", "lime", "lentils"],
     "compounds": ["tartaric acid", "citric acid", "malic acid", "furfural"],
     "dishes": ["sambar", "rasam", "pad thai", "imli chutney"]},
    {"ingredient": "ghee", "tastes": ["fatty_rich", "kokumi"], "cuisine": "Indian & South Asian",
     "partners": ["cumin", "cardamom", "garlic", "ginger", "rice", "lentils", "jaggery", "turmeric"],
     "compounds": ["butyric acid", "delta-decalactone", "pyrazines"],
     "dishes": ["biryani", "dal tadka", "halwa", "khichdi"]},
    {"ingredient": "curry leaves", "tastes": ["cooling", "bitter"], "cuisine": "Indian & South Asian",
     "partners": ["mustard seeds", "coconut", "tamarind", "turmeric", "chili", "lentils", "ginger"],
     "compounds": ["beta-caryophyllene", "linalool", "limonene"],
     "dishes": ["sambar", "coconut chutney", "upma", "rasam"]},
    {"ingredient": "kasuri methi", "tastes": ["bitter", "sweet"], "cuisine": "Indian & South Asian",
     "partners": ["tomato", "cream", "butter", "paneer", "garlic", "ginger", "garam masala"],
     "compounds": ["sotolon", "anethole", "coumarin"],
     "dishes": ["butter chicken", "paneer tikka masala", "dal makhani"]},
    {"ingredient": "jaggery", "tastes": ["sweet", "kokumi"], "cuisine": "Indian & South Asian",
     "partners": ["tamarind", "ginger", "cardamom", "coconut", "ghee", "sesame", "peanuts"],
     "compounds": ["sucrose", "pyrazines", "maltol", "furans"],
     "dishes": ["payasam", "sambar", "chikki", "masala chai"]},
    {"ingredient": "mustard seeds", "tastes": ["spicy_pungent", "bitter"], "cuisine": "Indian & South Asian",
     "partners": ["curry leaves", "turmeric", "coconut", "chili", "ginger", "potatoes", "vinegar"],
     "compounds": ["allyl isothiocyanate", "sinigrin"],
     "dishes": ["South Indian tadka", "pickles", "potato curry"]},

    # --- EAST & SOUTHEAST ASIAN ---
    {"ingredient": "ginger", "tastes": ["spicy_pungent", "cooling"], "cuisine": "East & Southeast Asian",
     "partners": ["garlic", "soy sauce", "scallion", "sesame oil", "chili", "lime", "honey", "cardamom"],
     "compounds": ["gingerol", "zingerone", "citral", "beta-sesquiphellandrene"],
     "dishes": ["stir-fry", "ramen broth", "chai", "ginger scallion chicken"]},
    {"ingredient": "soy sauce", "tastes": ["umami", "salty", "kokumi"], "cuisine": "East & Southeast Asian",
     "partners": ["ginger", "garlic", "sesame oil", "scallion", "honey", "rice vinegar", "mushroom"],
     "compounds": ["glutamic acid", "methional", "furaneol"],
     "dishes": ["teriyaki glaze", "stir-fry", "fried rice", "ramen"]},
    {"ingredient": "miso", "tastes": ["umami", "salty", "kokumi"], "cuisine": "East & Southeast Asian",
     "partners": ["ginger", "sesame oil", "mirin", "butter", "mushroom", "scallion", "honey", "eggplant"],
     "compounds": ["glutamic acid", "methional", "ethyl linoleate"],
     "dishes": ["miso soup", "miso glazed eggplant", "ramen", "miso butter pasta"]},
    {"ingredient": "fish sauce", "tastes": ["umami", "salty"], "cuisine": "East & Southeast Asian",
     "partners": ["lime", "chili", "garlic", "palm sugar", "lemongrass", "cilantro", "coconut milk"],
     "compounds": ["glutamic acid", "inosinate", "trimethylamine"],
     "dishes": ["pad thai", "nuoc cham", "Thai curry", "pho"]},
    {"ingredient": "sesame oil", "tastes": ["fatty_rich", "kokumi"], "cuisine": "East & Southeast Asian",
     "partners": ["soy sauce", "ginger", "garlic", "scallion", "rice vinegar", "chili", "tofu"],
     "compounds": ["2-furfurylthiol", "pyrazines", "sesamol", "guaiacol"],
     "dishes": ["sesame noodles", "fried rice", "dumpling dipping sauce"]},
    {"ingredient": "lemongrass", "tastes": ["cooling", "sour"], "cuisine": "East & Southeast Asian",
     "partners": ["coconut milk", "lime", "galangal", "chili", "garlic", "fish sauce", "ginger"],
     "compounds": ["citral", "myrcene", "geraniol"],
     "dishes": ["tom yum", "Thai green curry", "lemongrass chicken"]},
    {"ingredient": "coconut milk", "tastes": ["fatty_rich", "sweet"], "cuisine": "East & Southeast Asian",
     "partners": ["lime", "ginger", "lemongrass", "turmeric", "chili", "curry leaves", "mango"],
     "compounds": ["delta-octalactone", "gamma-nonalactone", "lauric acid"],
     "dishes": ["Thai green curry", "laksa", "Malabar stew", "mango sticky rice"]},
    {"ingredient": "sichuan peppercorn", "tastes": ["spicy_pungent", "cooling"], "cuisine": "East & Southeast Asian",
     "partners": ["dried chili", "garlic", "ginger", "soy sauce", "black vinegar", "sesame", "beef"],
     "compounds": ["hydroxy-alpha-sanshool", "limonene", "linalool"],
     "dishes": ["mapo tofu", "kung pao chicken", "dan dan noodles"]},
    {"ingredient": "shiitake mushroom", "tastes": ["umami", "kokumi"], "cuisine": "East & Southeast Asian",
     "partners": ["soy sauce", "ginger", "garlic", "sesame oil", "thyme", "butter", "scallion"],
     "compounds": ["guanylate", "lenthionine", "1-octen-3-ol", "glutamic acid"],
     "dishes": ["dashi broth", "stir-fry", "dumplings", "risotto"]},

    # --- MEDITERRANEAN & EUROPEAN ---
    {"ingredient": "garlic", "tastes": ["spicy_pungent", "umami", "kokumi"], "cuisine": "Mediterranean & European",
     "partners": ["lemon", "thyme", "butter", "olive oil", "rosemary", "ginger", "tomato", "basil"],
     "compounds": ["allicin", "diallyl disulfide", "allyl methyl sulfide"],
     "dishes": ["roasted chicken", "aglio e olio", "aioli", "dal tadka"]},
    {"ingredient": "lemon", "tastes": ["sour", "cooling"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "thyme", "butter", "olive oil", "rosemary", "parsley", "chicken", "honey"],
     "compounds": ["limonene", "citral", "citric acid", "gamma-terpinene"],
     "dishes": ["lemon herb chicken", "piccata", "ceviche", "lemon tart"]},
    {"ingredient": "thyme", "tastes": ["cooling", "bitter"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "lemon", "butter", "rosemary", "chicken", "mushroom", "shallot", "white wine"],
     "compounds": ["thymol", "carvacrol", "p-cymene", "linalool"],
     "dishes": ["roasted chicken", "beef bourguignon", "mushroom soup"]},
    {"ingredient": "rosemary", "tastes": ["cooling", "bitter"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "lemon", "thyme", "olive oil", "chicken", "lamb", "potatoes", "sea salt"],
     "compounds": ["1,8-cineole", "alpha-pinene", "camphor", "borneol"],
     "dishes": ["roasted chicken", "focaccia", "garlic roasted potatoes"]},
    {"ingredient": "butter", "tastes": ["fatty_rich", "kokumi"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "lemon", "thyme", "parsley", "shallot", "black pepper", "white wine", "sugar"],
     "compounds": ["diacetyl", "butyric acid", "delta-decalactone", "acetoin"],
     "dishes": ["pan-roasted chicken", "beurre blanc", "croissants", "mashed potatoes"]},
    {"ingredient": "olive oil", "tastes": ["fatty_rich", "bitter", "spicy_pungent"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "lemon", "thyme", "rosemary", "basil", "tomato", "oregano", "sea salt"],
     "compounds": ["hexanal", "oleocanthal", "linalool", "squalene"],
     "dishes": ["vinaigrette", "roasted vegetables", "hummus", "pasta"]},
    {"ingredient": "basil", "tastes": ["sweet", "cooling"], "cuisine": "Mediterranean & European",
     "partners": ["tomato", "garlic", "olive oil", "mozzarella", "parmesan", "lemon", "pine nuts", "strawberry"],
     "compounds": ["estragole", "linalool", "eugenol", "1,8-cineole"],
     "dishes": ["pesto", "caprese salad", "margherita pizza", "tomato soup"]},
    {"ingredient": "tomato", "tastes": ["umami", "sour", "sweet"], "cuisine": "Mediterranean & European",
     "partners": ["basil", "garlic", "olive oil", "oregano", "onion", "parmesan", "balsamic vinegar", "cumin"],
     "compounds": ["glutamic acid", "hexanal", "beta-ionone", "furaneol"],
     "dishes": ["marinara", "shakshuka", "bruschetta", "butter masala"]},
    {"ingredient": "parmesan", "tastes": ["umami", "salty", "fatty_rich", "kokumi"], "cuisine": "Mediterranean & European",
     "partners": ["garlic", "basil", "black pepper", "butter", "olive oil", "mushroom", "lemon", "tomato"],
     "compounds": ["butyric acid", "glutamic acid", "ethyl hexanoate", "isovaleric acid"],
     "dishes": ["risotto", "pesto", "cacio e pepe", "caesar salad"]},
    {"ingredient": "black pepper", "tastes": ["spicy_pungent"], "cuisine": "Universal Pantry",
     "partners": ["garlic", "lemon", "thyme", "butter", "olive oil", "parmesan", "turmeric", "strawberry"],
     "compounds": ["piperine", "beta-caryophyllene", "limonene", "sabinene"],
     "dishes": ["cacio e pepe", "steak au poivre", "roasted chicken"]},
    {"ingredient": "balsamic vinegar", "tastes": ["sour", "sweet", "astringent", "kokumi"], "cuisine": "Mediterranean & European",
     "partners": ["strawberry", "tomato", "olive oil", "basil", "parmesan", "fig", "rosemary"],
     "compounds": ["acetic acid", "furfural", "5-hydroxymethylfurfural", "tannins"],
     "dishes": ["caprese glaze", "roasted Brussels sprouts", "macerated berries"]},

    # --- LATIN AMERICAN & MIDDLE EASTERN ---
    {"ingredient": "lime", "tastes": ["sour", "cooling"], "cuisine": "Latin American",
     "partners": ["cilantro", "chili", "garlic", "ginger", "cumin", "fish sauce", "coconut milk", "avocado"],
     "compounds": ["limonene", "citral", "citric acid", "beta-pinene"],
     "dishes": ["guacamole", "ceviche", "tacos", "pad thai"]},
    {"ingredient": "cilantro", "tastes": ["cooling", "sour"], "cuisine": "Latin American",
     "partners": ["lime", "chili", "cumin", "garlic", "avocado", "tomato", "ginger", "coconut"],
     "compounds": ["(E)-2-decenal", "linalool", "decyl acetate"],
     "dishes": ["salsa", "guacamole", "green chutney", "pho"]},
    {"ingredient": "avocado", "tastes": ["fatty_rich"], "cuisine": "Latin American",
     "partners": ["lime", "cilantro", "chili", "tomato", "red onion", "sea salt", "garlic", "egg"],
     "compounds": ["oleic acid", "hexanal", "beta-caryophyllene"],
     "dishes": ["guacamole", "avocado toast", "poke bowls", "smoothies"]},
    {"ingredient": "smoked paprika", "tastes": ["spicy_pungent", "sweet", "kokumi"], "cuisine": "Latin American",
     "partners": ["garlic", "olive oil", "lemon", "cumin", "chickpeas", "potatoes", "chicken", "tomato"],
     "compounds": ["capsanthin", "guaiacol", "syringol", "pyrazines"],
     "dishes": ["patatas bravas", "paella", "hummus", "dry rubs"]},
    {"ingredient": "tahini", "tastes": ["fatty_rich", "bitter", "kokumi"], "cuisine": "Middle Eastern & North African",
     "partners": ["lemon", "garlic", "chickpeas", "cumin", "olive oil", "honey", "dark chocolate", "eggplant"],
     "compounds": ["sesamol", "pyrazines", "oleic acid"],
     "dishes": ["hummus", "baba ganoush", "halva", "tahini brownies"]},
    {"ingredient": "sumac", "tastes": ["sour", "astringent"], "cuisine": "Middle Eastern & North African",
     "partners": ["lemon", "olive oil", "garlic", "yogurt", "lamb", "chickpeas", "red onion", "za'atar"],
     "compounds": ["malic acid", "tannins", "beta-caryophyllene"],
     "dishes": ["fattoush", "kebabs", "hummus", "roasted onions"]},
    {"ingredient": "pomegranate", "tastes": ["sour", "sweet", "astringent"], "cuisine": "Middle Eastern & North African",
     "partners": ["walnut", "mint", "sumac", "yogurt", "eggplant", "lamb", "pistachio", "lemon"],
     "compounds": ["punicalagins", "citric acid", "anthocyanins", "tannins"],
     "dishes": ["fesenjan", "fattoush", "raita", "muhammara"]},

    # --- BAKING, PASTRY & SWEET-BITTER-COOLING ---
    {"ingredient": "dark chocolate", "tastes": ["bitter", "sweet", "fatty_rich", "kokumi"], "cuisine": "Baking & Pastry",
     "partners": ["espresso", "vanilla", "sea salt", "cinnamon", "chili", "orange", "hazelnut", "raspberry"],
     "compounds": ["pyrazines", "theobromine", "isovaleraldehyde", "phenylethyl alcohol"],
     "dishes": ["mocha mousse", "mole sauce", "brownies", "ganache"]},
    {"ingredient": "espresso", "tastes": ["bitter", "kokumi"], "cuisine": "Baking & Pastry",
     "partners": ["dark chocolate", "vanilla", "cinnamon", "cardamom", "butter", "brown sugar", "cream"],
     "compounds": ["2-furfurylthiol", "pyrazines", "guaiacol", "vanillin"],
     "dishes": ["tiramisu", "mocha brownies", "coffee steak rub", "affogato"]},
    {"ingredient": "cinnamon", "tastes": ["sweet", "spicy_pungent"], "cuisine": "Baking & Pastry",
     "partners": ["apple", "cardamom", "vanilla", "nutmeg", "clove", "honey", "dark chocolate", "cumin"],
     "compounds": ["cinnamaldehyde", "eugenol", "linalool"],
     "dishes": ["apple pie", "chai", "moroccan tagine", "cinnamon rolls"]},
    {"ingredient": "vanilla", "tastes": ["sweet", "kokumi"], "cuisine": "Baking & Pastry",
     "partners": ["dark chocolate", "espresso", "cinnamon", "strawberry", "banana", "butter", "cream", "cardamom"],
     "compounds": ["vanillin", "p-hydroxybenzaldehyde", "vanillic acid"],
     "dishes": ["custard", "banana shake", "cookies", "ice cream"]},
    {"ingredient": "banana", "tastes": ["sweet", "fatty_rich", "astringent"], "cuisine": "Baking & Pastry",
     "partners": ["cinnamon", "vanilla", "dark chocolate", "peanut butter", "oats", "honey", "cardamom", "walnut"],
     "compounds": ["isoamyl acetate", "eugenol", "pectin", "tannins"],
     "dishes": ["banana shake", "banana bread", "pancakes", "nice cream"]},
    {"ingredient": "mint", "tastes": ["cooling", "sweet"], "cuisine": "Universal Pantry",
     "partners": ["lime", "cucumber", "yogurt", "lamb", "dark chocolate", "watermelon", "peas", "ginger"],
     "compounds": ["menthol", "carvone", "menthone", "limonene"],
     "dishes": ["mint chutney", "mojito", "raita", "mint chocolate"]},
    {"ingredient": "black tea", "tastes": ["astringent", "bitter"], "cuisine": "Universal Pantry",
     "partners": ["cardamom", "ginger", "milk", "lemon", "honey", "cinnamon", "bergamot"],
     "compounds": ["theaflavins", "thearubigins", "caffeine", "linalool"],
     "dishes": ["masala chai", "iced tea", "tea-smoked duck"]},
    {"ingredient": "sea salt", "tastes": ["salty"], "cuisine": "Universal Pantry",
     "partners": ["dark chocolate", "butter", "caramel", "olive oil", "watermelon", "tomato", "lemon"],
     "compounds": ["sodium chloride", "magnesium chloride"],
     "dishes": ["salted caramel", "focaccia", "chocolate chip cookies"]},
]


def merge_local_recipes(recipes_path: Path, existing: dict[str, dict], max_extra: int = 400) -> None:
    if not recipes_path.exists():
        return
    freq: Counter[str] = Counter()
    cooccur: dict[str, Counter[str]] = defaultdict(Counter)
    dishes: dict[str, list[str]] = defaultdict(list)

    with open(recipes_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (rec.get("title") or "").strip()
            raw = rec.get("ner") or rec.get("ingredients") or []
            if not isinstance(raw, list):
                continue
            items = sorted({str(x).strip().lower() for x in raw if 2 <= len(str(x).strip()) <= 30})
            for item in items:
                freq[item] += 1
                if title and len(dishes[item]) < 4 and title not in dishes[item]:
                    dishes[item].append(title)
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    cooccur[a][b] += 1
                    cooccur[b][a] += 1

    added = 0
    for ing, count in freq.most_common(max_extra + len(existing)):
        partners = [p for p, _ in cooccur[ing].most_common(8)]
        if ing in existing:
            for p in partners:
                if p not in existing[ing]["partners"] and len(existing[ing]["partners"]) < 12:
                    existing[ing]["partners"].append(p)
        elif added < max_extra:
            existing[ing] = {
                "ingredient": ing,
                "tastes": ["umami"],
                "cuisine": "Universal Pantry",
                "partners": partners,
                "compounds": [],
                "dishes": dishes[ing],
            }
            added += 1


def main() -> None:
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 1. Rebuild flavor_substitutions cleanly
    try:
        client.delete_collection("flavor_substitutions")
    except Exception:
        pass
    sub_col = client.create_collection("flavor_substitutions", metadata={"hnsw:space": "cosine"})

    sub_ids, sub_docs, sub_metas = [], [], []
    for r in UNIVERSAL_SUBSTITUTIONS:
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

    # 2. Rebuild flavor_pairings cleanly
    try:
        client.delete_collection("flavor_pairings")
    except Exception:
        pass
    pair_col = client.create_collection("flavor_pairings", metadata={"hnsw:space": "cosine"})

    ing_map = {item["ingredient"].lower(): dict(item) for item in UNIVERSAL_INGREDIENTS}
    merge_local_recipes(RAW_DIR / "recipes_25k.jsonl", ing_map, max_extra=400)

    pair_ids, pair_docs, pair_metas = [], [], []
    for idx, (ing, data) in enumerate(ing_map.items()):
        tastes_str = ", ".join(data["tastes"])
        lines = [
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

    print(f"Indexed {sub_col.count()} universal substitution rules.")
    print(f"Indexed {pair_col.count()} universal ingredient taste profiles.")


if __name__ == "__main__":
    main()
