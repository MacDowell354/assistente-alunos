import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"
CHUNK_SIZE = 1000

def build_index():
    # 1) Inicializa o client com nova API ChromaDB
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=INDEX_DIR
    ))

    # 2) Função de embedding OpenAI
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    # 3) Cria ou obtém a coleção
    collection = client.get_or_create_collection(
        name="transcripts",
        embedding_function=embed_fn
    )

    # 4) Lê e divide em chunks
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 5) Upsert
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{} for _ in chunks]
    )

    # 6) Persiste
    client.persist()
    print(f"✅ Índice gerado em '{INDEX_DIR}' com {len(chunks)} chunks.")

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    build_index()
