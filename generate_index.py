# generate_index.py

import os
import chromadb
from chromadb.utils import embedding_functions

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"

def main():
    # 1) inicializa o client com a nova API
    client = chromadb.Client(
        chroma_db_impl="duckdb+parquet",
        persist_directory=INDEX_DIR
    )

    # 2) prepara a função de embedding da OpenAI
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    # 3) cria ou obtém a coleção
    collection = client.get_or_create_collection(
        name="transcripts",
        embedding_function=embed_fn
    )

    # 4) lê todo o texto e chunkiza
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 5) adiciona (overwrite) os chunks na coleção
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{} for _ in chunks]
    )

    # 6) persiste em disco
    client.persist()
    print(f"✅ Índice gerado em {INDEX_DIR} com {len(chunks)} chunks.")

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    main()
