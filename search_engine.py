import os
from llama_index import (
    load_index_from_storage,
    StorageContext,
    OpenAIEmbedding,
    ServiceContext,
)

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# configura o embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# carrega o índice já gerado na pasta ./storage
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context, service_context=service_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta,
    usando o índice carregado em memória.
    """
    q_engine = index.as_query_engine()
    result = q_engine.query(question)
    return str(result)
