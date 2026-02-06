from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys

# -------- CONFIG --------
INDEX_DIR = "/home/achilles/Dev/index"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
TOP_K = 6
# ------------------------

def main(query: str):
    print("Setting up embeddings and storage context...")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        trust_remote_code=True,
    )

    Settings.llm = None  # 🔥 THIS IS THE KEY LINE

    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_DIR
    )

    print("Reconstructing index...")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        llm=None,
    )

    print("Querying index...")
    response = query_engine.query(query)

    print("\n=== RETRIEVED CONTEXT ===\n")
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        print(f"[{i}] {meta.get('file_path', 'unknown')}")
        print("-" * 60)
        print(node.node.text[:800])
        print()

if __name__ == "__main__":
    main(sys.argv[1])
