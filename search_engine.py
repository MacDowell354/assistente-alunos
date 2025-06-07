import os

from llama_index import load_index_from_storage, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Mesma configuração de embedding usada no build
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# Carrega o índice já persistido em disco
storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_ctx, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Executa query semântica sobre o índice e devolve
    o trecho mais relevante.
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
