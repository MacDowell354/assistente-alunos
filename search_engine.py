import os
from llama_index import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# inicializa embedding
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# carrega índice de disco
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """Retorna o trecho mais relevante do índice para a pergunta."""
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
