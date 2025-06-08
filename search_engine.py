import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage/chroma"
TOP_K = 3

# 1) Inicializa o client com os mesmos Settings
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=INDEX_DIR
))

# 2) Mesma função de embedding
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# 3) Obtém a coleção já criada
collection = client.get_collection(
    name="transcripts",
    embedding_function=embed_fn
)

def retrieve_relevant_context(question: str) -> str:
    """
    Consulta os TOP_K chunks mais relevantes para a pergunta
    e retorna a concatenação deles como contexto.
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )
    docs = results["documents"][0]  # lista de strings
    # junta com separador para clareza
    return "\n\n---\n\n".join(docs)
