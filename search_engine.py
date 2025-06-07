import os
from llama_index import (
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    OpenAIEmbedding,
)

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# --- Mesmo ServiceContext do generate_index ---
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
)

# --- Carrega o índice já persistido em disco ---
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o trecho mais relevante do índice
    para a pergunta recebida.
    """
    q_engine = index.as_query_engine()
    resp = q_engine.query(question)
    return str(resp)
