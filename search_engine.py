import os
from llama_index import StorageContext, load_index_from_storage
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

# --- Carrega o índice (persistido em ./storage) ---
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta,
    usando o índice carregado em memória.
    """
    q_engine = index.as_query_engine()
    response = q_engine.query(question)
    return str(response)
