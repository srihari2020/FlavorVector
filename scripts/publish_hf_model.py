import json
import shutil
from pathlib import Path
import chromadb
from huggingface_hub import HfApi, get_token, login
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "hf_model_dist"
FENCE = "`" * 3

INCLUDE_DIRS = ["app", "scripts", "tests", "assets", "chroma_db", "data"]
INCLUDE_FILES = [
    "streamlit_app.py",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
]


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

**FlavorVector** is a standalone, local-first semantic cooking substitution, flavor-pairing, and pantry flavor-bridge engine grounded in food-science chemistry and volatile aroma compounds. This repository includes the complete codebase, Streamlit UI, FastAPI service, pre-indexed ChromaDB database, and bundled transformer weights for offline execution.

## Repository Contents
- `models/bi_encoder/` — Bundled `BAAI/bge-base-en-v1.5` dense retrieval model (`~438 MB`).
- `models/cross_encoder/` — Bundled `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (`~91 MB`).
- `chroma_db/` — Pre-built persistent vector store ({atlas['total_vectors']} vectors across `flavor_substitutions` and `flavor_pairings`).
- `flavor_atlas.json` — Exported structured food-science substitution rules and ingredient compound profiles.
- `app/` — Core retrieval engine (`engine.py`), live web synthesis (`live_fetcher.py`), ratio scaler (`balancer.py`), and FastAPI endpoints (`main.py`).
- `streamlit_app.py` — Interactive Streamlit dashboard with 10-taste Plotly radar charts.
- `scripts/` & `tests/` — Atlas builders, 25-case benchmark runner, PNG chart generator, and `pytest` suite.

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

## Standalone Usage

{FENCE}python
import sys
from huggingface_hub import snapshot_download

# Downloads the full repo (bundled bi-encoder, cross-encoder, chroma_db, and app)
local_dir = snapshot_download(repo_id="{repo_id}", repo_type="model")
sys.path.insert(0, local_dir)

from app.engine import FlavorEngine

engine = FlavorEngine()

subs, _ = engine.search_substitutes(
    query="what can replace buttermilk and give the same tanginess in pancakes?",
    top_k=3,
)
for hit in subs:
    print(f"#{{hit['rank']}} ({{hit['relevance_score']*100:.1f}}%): {{hit['substitute']}}")
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

    # 1. Save full Bi-Encoder and Cross-Encoder model weights locally
    print("Saving BAAI/bge-base-en-v1.5 weights (~438 MB)...")
    bi_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    bi_model.save(str(DIST_DIR / "models" / "bi_encoder"))

    print("Saving cross-encoder/ms-marco-MiniLM-L-6-v2 weights (~91 MB)...")
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    ce_model.save(str(DIST_DIR / "models" / "cross_encoder"))

    # 2. Copy all project directories and root files
    ignore_fn = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    for folder in INCLUDE_DIRS:
        src = PROJECT_ROOT / folder
        if src.exists():
            shutil.copytree(src, DIST_DIR / folder, ignore=ignore_fn)

    for fname in INCLUDE_FILES:
        src = PROJECT_ROOT / fname
        if src.exists():
            shutil.copy2(src, DIST_DIR / fname)

    # 3. Export JSON snapshot of all indexed ChromaDB collections
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

    # 4. Generate Model Card README.md
    bench = json.loads((PROJECT_ROOT / "tests" / "full_benchmark.json").read_text(encoding="utf-8"))
    (DIST_DIR / "README.md").write_text(build_model_card(bench, repo_id), encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"Uploading complete {total_mb:.1f} MB bundle to {repo_id}...")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(DIST_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload complete FlavorVector bundle (model weights, ChromaDB, UI, API, scripts & tests)",
    )

    print(f"\nDone! Full repository uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
