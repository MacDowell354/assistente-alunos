import os

from llama_index import ServiceContext
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage import StorageContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa o serviço de embedding
embed = OpenAIEmbedding(api_key=OPENAI_API_KEY)
service_ctx = ServiceContext.from_defaults(embed_model=embed)

# Carrega o índice da pasta ./storage
storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = GPTVectorStoreIndex.load_from_storage(storage_ctx, service_context=service_ctx)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o trecho mais relevante do índice
    para a pergunta dada.
    """
    q_engine = index.as_query_engine()
    resp = q_engine.query(question)
    return str(resp)
