import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.engine import FlavorEngine  # noqa: E402

BENCHMARK_CASES = [
    {"query": "what can replace buttermilk and give the same tanginess in pancakes?", "vegan_only": False, "expected_any": ["lemon juice", "yogurt"]},
    {"query": "vegan substitute for buttermilk in cornbread", "vegan_only": True, "expected_any": ["soy milk", "apple cider vinegar"]},
    {"query": "replace egg as a binder in meatballs", "vegan_only": True, "expected_any": ["flaxseed"]},
    {"query": "replace egg for airy lift and leavening in sponge cake", "vegan_only": True, "expected_any": ["baking soda"]},
    {"query": "vegan substitute for whipped egg whites in meringue", "vegan_only": True, "expected_any": ["aquafaba"]},
    {"query": "what can I use instead of sugar in a banana shake?", "vegan_only": True, "expected_any": ["dates", "maple syrup", "jaggery"]},
    {"query": "unrefined Indian sweetener replacement for brown sugar in kheer", "vegan_only": True, "expected_any": ["jaggery"]},
    {"query": "substitute for tamarind paste in sambar", "vegan_only": True, "expected_any": ["amchur", "lime juice"]},
    {"query": "what replaces kokum in coastal fish curry?", "vegan_only": True, "expected_any": ["tamarind", "pomegranate"]},
    {"query": "dry citrus spice alternative for ground sumac in fattoush", "vegan_only": True, "expected_any": ["lemon zest", "citric acid"]},
    {"query": "vegan substitute for fish sauce umami in stir-fry", "vegan_only": True, "expected_any": ["soy sauce", "nori", "mushroom"]},
    {"query": "vegetarian replacement for oyster sauce in noodles", "vegan_only": True, "expected_any": ["mushroom stir-fry", "hoisin"]},
    {"query": "what can replace miso paste in ramen broth?", "vegan_only": True, "expected_any": ["tahini", "nutritional yeast"]},
    {"query": "dairy-free substitute for grated parmesan cheese on pasta", "vegan_only": True, "expected_any": ["nutritional yeast"]},
    {"query": "vegan replacement for anchovy fillets in puttanesca sauce", "vegan_only": True, "expected_any": ["capers", "miso"]},
    {"query": "substitute for asafoetida hing in dal tadka", "vegan_only": True, "expected_any": ["garlic powder", "onion powder"]},
    {"query": "what can replace gochujang chili paste in bibimbap?", "vegan_only": True, "expected_any": ["red miso", "gochugaru", "cayenne"]},
    {"query": "substitute for chipotle in adobo in tacos", "vegan_only": True, "expected_any": ["smoked paprika"]},
    {"query": "what replaces kasuri methi dried fenugreek leaves in butter masala?", "vegan_only": True, "expected_any": ["celery leaves", "fennel"]},
    {"query": "substitute for fresh curry leaves in South Indian tadka", "vegan_only": True, "expected_any": ["bay leaf", "lime zest"]},
    {"query": "what can I use instead of coconut chutney with masala dosa?", "vegan_only": True, "expected_any": ["peanut", "chana dal", "kara chutney"]},
    {"query": "vegan substitute for paneer in palak paneer", "vegan_only": True, "expected_any": ["tofu"]},
    {"query": "dairy-free heavy cream substitute for tomato gravy", "vegan_only": True, "expected_any": ["cashews", "coconut cream"]},
    {"query": "glossy sauce thickener replacement for cornstarch", "vegan_only": True, "expected_any": ["arrowroot", "potato starch"]},
    {"query": "vegan meaty substitute for boneless chicken in biryani", "vegan_only": True, "expected_any": ["soya chunks", "jackfruit"]},
]


def run_benchmark() -> None:
    engine = FlavorEngine()
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []
    latencies_ms = []
    detailed = []

    for case in BENCHMARK_CASES:
        t0 = time.perf_counter()
        results, _ = engine.search_substitutes(
            query=case["query"],
            vegan_only=case["vegan_only"],
            force_live=False,
            top_k=3,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        found_rank = 0
        for idx, hit in enumerate(results, start=1):
            sub_lower = hit["substitute"].lower()
            if any(kw.lower() in sub_lower for kw in case["expected_any"]):
                found_rank = idx
                break

        if found_rank == 1:
            hit_at_1 += 1
        if 1 <= found_rank <= 3:
            hit_at_3 += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        detailed.append(
            {
                "query": case["query"],
                "found_rank": found_rank,
                "latency_ms": round(elapsed_ms, 2),
                "top_hit": results[0]["substitute"] if results else None,
            }
        )

    total = len(BENCHMARK_CASES)
    summary = {
        "device": engine.device,
        "total_cases": total,
        "hit_rate_at_1": round(hit_at_1 / total * 100, 1),
        "hit_rate_at_3": round(hit_at_3 / total * 100, 1),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "avg_latency_ms": round(sum(latencies_ms) / total, 2),
        "cases": detailed,
    }

    out_path = PROJECT_ROOT / "tests" / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FlavorVector Retrieval Benchmark ({engine.device.upper()}) ===")
    print(f"Total Test Cases : {total}")
    print(f"Hit Rate@1       : {summary['hit_rate_at_1']}%")
    print(f"Hit Rate@3       : {summary['hit_rate_at_3']}%")
    print(f"MRR              : {summary['mrr']}")
    print(f"Avg Query Time   : {summary['avg_latency_ms']} ms")
    print(f"Saved report to  : {out_path}\n")


if __name__ == "__main__":
    run_benchmark()
