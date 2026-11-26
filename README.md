# FlavorVector

A local-first, self-expanding **Semantic Cooking Substitution & Flavor-Pairing Engine** powered by food-science chemistry, volatile aroma compounds, and two-stage neural retrieval.

## Why FlavorVector?
Standard keyword search matches ingredient names rather than **culinary and chemical function**—suggesting heavy cream as a substitute for buttermilk because both are dairy, even though heavy cream lacks the acidity required to activate baking soda in pancakes. FlavorVector embeds structured **role-first food-science cards** across **10 taste & mouthfeel dimensions** and **7 global culinary regions**, reranks candidates with a Cross-Encoder, and falls back to real-time web knowledge synthesis on cache misses.

## Architecture
- **Dense Bi-Encoder Retrieval:** `BAAI/bge-base-en-v1.5` over persistent `ChromaDB` cosine collections (`flavor_substitutions` & `flavor_pairings`)
- **Cross-Encoder Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` with sigmoid confidence calibration
- **10-Taste Taxonomy:** Sweet, Salty, Sour, Bitter, Umami, Spicy/Pungent, Astringent, Cooling, Fatty/Rich, and Kokumi
- **Hybrid Real-Time Learning:** Queries Wikipedia, TheMealDB, Open Food Facts, and local Ollama on low-confidence misses and auto-upserts new vectors into `ChromaDB`
- **Interfaces:** Interactive `Streamlit` UI (with 10-taste Plotly radar charts) + `FastAPI` REST service

## Retrieval Benchmark (25 Culinary Edge Cases)
| Metric | Score |
| :--- | :--- |
| **Compute Device** | `CUDA` |
| **Hit Rate@1** | **100.0%** |
| **Hit Rate@3** | **100.0%** |
| **Mean Reciprocal Rank (MRR)** | **1.0** |
| **Avg Query Latency** | **84.76 ms** |

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Index the Universal 10-Taste & Global Cuisine Atlas
python3 scripts/expand_global_atlas.py

# 2. Run the Streamlit UI (http://localhost:8501)
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

# 3. Or run the FastAPI REST API (http://localhost:8000/docs)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
