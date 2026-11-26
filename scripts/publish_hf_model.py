import json
import shutil
from pathlib import Path
import chromadb
from huggingface_hub import HfApi, get_token, login

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "hf_model_dist"
FENCE = "`" * 3


def build_model_card(bench: dict, repo_id: str) -> str:
    acc = bench["accuracy"]["two_stage"]
    bi_acc = bench["accuracy"]["bi_encoder"]
    lat = bench["latency"]
    qps = bench["throughput_qps"]
    sys_info = bench["system"]
    atlas = bench["atlas"]

    return f"""---
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model:
  - BAAI/bge-base-en-v1.5
  - cross-encoder/ms-marco-MiniLM-L-6-v2
tags:
  - sentence-transformers
  - chromadb
  - semantic-search
  - reranking
  - food-science
  - flavor-pairing
  - culinary-ai
  - information-retrieval
---

# FlavorVector

![FlavorVector Benchmarks](assets/benchmark_dashboard.png)

![FlavorVector 10-Taste & Cuisine Coverage](assets/taste_coverage.png)

**FlavorVector** is a local-first, two-stage semantic cooking substitution, flavor-pairing, and pantry flavor-bridge engine grounded in food-science chemistry and volatile aroma compounds.

## Model & Index Architecture
- **Stage 1 — Dense Bi-Encoder Retrieval:** `BAAI/bge-base-en-v1.5` over pre-indexed `ChromaDB` cosine collections (`flavor_substitutions` and `flavor_pairings`).
- **Stage 2 — Cross-Encoder Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` with sigmoid confidence calibration.
- **10-Taste & Mouthfeel Taxonomy:** `sweet`, `salty`, `sour`, `bitter`, `umami`, `spicy_pungent`, `astringent`, `cooling`, `fatty_rich`, and `kokumi` across 7 global culinary regions.
- **Included Artifacts:**
  - `chroma_db/` — Pre-built persistent vector database ({atlas['total_vectors']} vectors) ready for zero-build inference.
  - `flavor_atlas.json` — Exported structured food-science substitution rules and ingredient compound profiles.
  - `app/` — Modular Python retrieval engine (`FlavorEngine`), kitchen fraction scaler (`balancer.py`), and live web knowledge fetcher (`live_fetcher.py`).

## Evaluation Benchmarks
Benchmarked across **25 culinary edge cases** on `{sys_info['gpu_name'] or sys_info['device'].upper()}`:

| Category | Metric | Stage 1 (`bge-base-en-v1.5`) | Two-Stage (`+ Cross-Encoder`) |
| :--- | :--- | :---: | :---: |
| **Accuracy** | **Hit Rate@1** | {bi_acc['hit_at_1']}% | **{acc['hit_at_1']}%** |
| **Accuracy** | **Hit Rate@3** | {bi_acc['hit_at_3']}% | **{acc['hit_at_3']}%** |
| **Ranking Quality** | **Mean Reciprocal Rank (MRR)** | {bi_acc['mrr']}% | **{acc['mrr']}%** |
| **Ranking Quality** | **NDCG@3** | {bi_acc['ndcg_at_3']}% | **{acc['ndcg_at_3']}%** |
| **Latency** | **Substitutions (p50 / p95)** | {lat['sub_stage1_mean_ms']} ms | **{lat['sub_total_p50_ms']} ms / {lat['sub_total_p95_ms']} ms** |
| **Latency** | **Flavor Pairing (p50 / p95)** | — | **{lat['pair_p50_ms']} ms / {lat['pair_p95_ms']} ms** |
| **Latency** | **Pantry Flavor Bridge (p50 / p95)** | — | **{lat['bridge_p50_ms']} ms / {lat['bridge_p95_ms']} ms** |
| **Throughput** | **Sustained QPS** | — | **{qps['substitutions']} QPS** (`/substitute`) · **{qps['pairings']} QPS** (`/pair`) |

## Usage with `huggingface_hub`

{FENCE}python
import sys
from huggingface_hub import snapshot_download

# 1. Download the pre-indexed ChromaDB store and retrieval engine
local_dir = snapshot_download(repo_id="{repo_id}", repo_type="model")
sys.path.insert(0, local_dir)

from app.engine import FlavorEngine

# 2. Load the two-stage engine (uses pre-built chroma_db automatically)
engine = FlavorEngine()

# 3. Search role-matched cooking substitutions
subs, _ = engine.search_substitutes(
    query="what can replace buttermilk and give the same tanginess in pancakes?",
    top_k=3,
)
for hit in subs:
    print(f"#{{hit['rank']}} ({{hit['relevance_score']*100:.1f}}%): {{hit['substitute']}}")

# 4. Find complementary flavor pairings
pairings, _ = engine.suggest_pairings(["garlic", "lemon", "thyme"], top_k=5)
for p in pairings:
    print(f"#{{p['rank']}}: {{p['ingredient']}} -> {{p['compounds']}}")

# 5. Solve missing bridge ingredients for what is in your pantry
bridges = engine.find_flavor_bridges(["tomato", "garlic", "paneer"], top_k=3)
for b in bridges:
    print(f"Bridge: {{b['ingredient']}} (links {{b['bridge_coverage']}} items)")
{FENCE}
"""


def main() -> None:
    if not get_token():
        login()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/FlavorVector"

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    # 1. Copy pre-built ChromaDB, benchmark PNGs, and inference package
    shutil.copytree(PROJECT_ROOT / "chroma_db", DIST_DIR / "chroma_db")
    shutil.copytree(PROJECT_ROOT / "assets", DIST_DIR / "assets")
    shutil.copytree(
        PROJECT_ROOT / "app",
        DIST_DIR / "app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(PROJECT_ROOT / "requirements.txt", DIST_DIR / "requirements.txt")

    # 2. Export clean JSON dataset snapshot from ChromaDB collections
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
    sub_data = client.get_collection("flavor_substitutions").get(include=["metadatas", "documents"])
    pair_data = client.get_collection("flavor_pairings").get(include=["metadatas", "documents"])

    atlas_export = {
        "substitutions": [
            {"id": i, **m, "document": d}
            for i, m, d in zip(sub_data["ids"], sub_data["metadatas"], sub_data["documents"])
        ],
        "ingredient_pairings": [
            {"id": i, **m, "document": d}
            for i, m, d in zip(pair_data["ids"], pair_data["metadatas"], pair_data["documents"])
        ],
    }
    (DIST_DIR / "flavor_atlas.json").write_text(json.dumps(atlas_export, indent=2), encoding="utf-8")

    # 3. Generate Model Card README.md
    bench = json.loads((PROJECT_ROOT / "tests" / "full_benchmark.json").read_text(encoding="utf-8"))
    (DIST_DIR / "README.md").write_text(build_model_card(bench, repo_id), encoding="utf-8")

    # 4. Create Model repository (not Space) and upload artifacts
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(DIST_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload FlavorVector pre-indexed ChromaDB, food-science atlas, and benchmarks",
    )

    print(f"\nPublished Hugging Face Model Repository: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
