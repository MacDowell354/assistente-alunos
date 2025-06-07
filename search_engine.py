import os

# contexto de storage para carregar
from llama_index.storage.storage_context import StorageContext
# helper para carregar o índice
from llama_index import load_index_from_storage
# embeddings e ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.service_context import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# --- Mesma configuração de embedding do generate_index ---
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# --- Carrega o índice persistido em ./storage ---
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta
    usando o índice carregado em memória.
    """
    q_engine = index.as_query_engine()
    resp = q_engine.query(question)
    return str(resp)
