import os
from llama_index import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# inicializa embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# carrega contexto de armazenamento e índice
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(
    storage_context,
    service_context=ServiceContext.from_defaults(embed_model=embed_model)
)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o trecho mais relevante da transcrição
    para enriquecer a pergunta ao GPT.
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)
