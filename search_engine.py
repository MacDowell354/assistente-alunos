# search_engine.py
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"
TOP_K = 3  # quantos pedaços trazer

# 1) inicializa client e coleção (mesmo nome do generate_index.py)
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=INDEX_DIR
    )
)
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = client.get_collection(
    name="transcripts",
    embedding_function=embed_fn
)

def retrieve_relevant_context(question: str) -> str:
    """
    Consulta os TOP_K pedaços mais relevantes para 'question'
    e retorna a concatenação deles.
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )
    # results["documents"] é lista de listas, pegamos a primeira
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs)
