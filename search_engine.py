import os
from llama_index import load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Configura o serviço de embeddings (mesmo usado na geração do índice)
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# Carrega o índice desde ./storage/index.json
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta,
    buscando no índice já carregado.
    """
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query(question)
    return str(response)
