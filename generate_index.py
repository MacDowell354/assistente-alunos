# generate_index.py
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- Carrega config & chave ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"

def main():
    # 1) prepara o client ChromaDB com parquet persistido
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=INDEX_DIR
        )
    )

    # 2) cria ou obtém a coleção, apontando a função de embedding da OpenAI
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name="transcripts",
        embedding_function=embed_fn
    )

    # 3) lê e chunkiza o arquivo inteiro
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # divide em pedaços de ~1000 caracteres
    chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 4) adiciona à coleção
    #    (o próprio ChromaDB vai chamar embed_fn internamente)
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=[{} for _ in chunks]
    )

    # 5) persiste no disco
    client.persist()
    print(f"✅ Índice gerado em {INDEX_DIR} com {len(chunks)} chunks.")

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    main()
