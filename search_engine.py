import os
from llama_index import (
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext, LLMPredictor
from llama_index import QueryMode

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa modelo de embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm_predictor=LLMPredictor()
)

# Carrega índice
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para 'question',
    usando o índice carregado em storage/.
    """
    query_engine = index.as_query_engine(
        service_context=service_context,
        response_mode=QueryMode.EMBEDDING
    )
    response = query_engine.query(question)
    return str(response)
