
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

import os

# -------- CONFIG --------
REPO_PATH = "/home/achilles/Documents/blog"     # change if needed
INDEX_DIR = "/home/achilles/Dev/index"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# Chunking tuned for code
Settings.chunk_size = 500
Settings.chunk_overlap = 50

# Embeddings (CPU)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    trust_remote_code=True
)
# ------------------------

def main():
    print("Loading documents...")
    documents = SimpleDirectoryReader(
        input_dir=REPO_PATH,
        recursive=True,
        filename_as_id=True,
        required_exts=[
            ".py", ".js", ".jsx", ".ts", ".tsx",
            ".json", ".yml", ".yaml",
            ".md", ".html", ".css"
        ],
        exclude_hidden=True,
        exclude=[
            "**/venv/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
        ],
    ).load_data()


    print(f"Loaded {len(documents)} documents")

    # FAISS index
    dimension = 768  # nomic embeddings
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    print("Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
    )

    print("Persisting index...")
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print(f"Indexed {len(documents)} documents")
    print("✅ Index built successfully")

if __name__ == "__main__":
    main()
