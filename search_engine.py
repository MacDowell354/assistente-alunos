import os
from llama_index import load_index_from_storage, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# inicializa o embedding
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# carrega o índice da pasta ./storage
storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
service_ctx = ServiceContext.from_defaults()
index = load_index_from_storage(storage_ctx, service_context=service_ctx)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta,
    usando o índice carregado em memória.
    """
    q_engine = index.as_query_engine()
    result = q_engine.query(question)
    return str(result)
