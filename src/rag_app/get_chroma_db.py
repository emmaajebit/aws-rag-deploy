from pathlib import Path

from langchain_community.vectorstores import Chroma
from src.rag_app.get_embeddings import get_embedding_function

# Resolved relative to this file so it matches create_db.py (src/data/chroma)
# regardless of the working directory (repo root, src/, or Lambda task root).
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
CHROMA_DB_INSTANCE = None  # Reference to singleton instance of ChromaDB


def get_chroma_db():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:
        # Prepare the DB.
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
        print(f"✅ Init ChromaDB {CHROMA_DB_INSTANCE} from {CHROMA_PATH}")

    return CHROMA_DB_INSTANCE