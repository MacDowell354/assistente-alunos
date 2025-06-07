import os
from llama_index import StorageContext, load_index_from_storage, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "./storage"

# Cria context de serviço (mesmo de antes) para usar embeddings
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

# Carrega o índice já persistido
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Faz query semântica no índice e retorna o texto mais relevante.
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
