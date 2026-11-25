import json
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def format_semantic_card(rule: dict) -> str:
    return (
        f"Replaces: {rule['target']}\n"
        f"Substitute: {rule['substitute']}\n"
        f"Culinary & Chemical Role: {rule['role']}\n"
        f"Flavor Profile: {rule['flavor_profile']}\n"
        f"Key Flavor & Functional Compounds: {', '.join(rule['compounds'])}\n"
        f"Best Dish Contexts: {', '.join(rule['best_in'])}\n"
        f"Avoid In: {', '.join(rule['avoid_in'])}"
    )


def main() -> None:
    seed_path = RAW_DIR / "substitutions_seed.json"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Missing {seed_path}. Run `python3 scripts/fetch_data.py` first."
        )

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="flavor_substitutions",
        metadata={"hnsw:space": "cosine"},
    )

    with open(seed_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    ids = [r["id"] for r in rules]
    documents = [format_semantic_card(r) for r in rules]
    metadatas = [
        {
            "target": r["target"],
            "substitute": r["substitute"],
            "vegan": r["vegan"],
        }
        for r in rules
    ]

    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(ids)} functional substitution cards into {CHROMA_DIR}\n")

    test_query = "what can replace buttermilk and give the same tanginess in pancakes?"
    q_vec = model.encode([test_query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=2)

    print(f"Query: {test_query}\n")
    for rank, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]), start=1):
        print(f"--- Rank #{rank} (cosine distance: {dist:.4f}) ---")
        print(doc + "\n")


if __name__ == "__main__":
    main()
