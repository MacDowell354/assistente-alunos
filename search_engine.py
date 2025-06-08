# search_engine.py

import os
import chromadb
from chromadb.utils import embedding_functions

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"
TOP_K = 3

# 1) inicializa client (mesmo em generate_index)
client = chromadb.Client(
    chroma_db_impl="duckdb+parquet",
    persist_directory=INDEX_DIR
)

# 2) mesma função de embedding
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# 3) obtém a coleção existente
collection = client.get_collection(
    name="transcripts",
    embedding_function=embed_fn
)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna os TOP_K chunks mais relevantes para a pergunta,
    concatenados como contexto.
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )
    docs = results["documents"][0]  # lista de strings
    # separa por -- para visualização clara
    return "\n\n---\n\n".join(docs)
