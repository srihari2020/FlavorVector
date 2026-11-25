# FlavorVector

A local-first **Semantic Cooking Substitution & Flavor-Pairing Engine** powered by food-science chemistry, volatile flavor compounds, and recipe co-occurrence.

## Stack
- **Embeddings & Reranking:** `SentenceTransformers` (`BAAI/bge-base-en-v1.5`)
- **Vector Store:** `ChromaDB` (Persistent local storage)
- **Backend API:** `FastAPI` + `Uvicorn`
