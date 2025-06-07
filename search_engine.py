import os
from llama_index import load_index_from_storage, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa o embedding (pode ser omitido se já embutido no StorageContext)
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Carrega o índice da pasta de storage
storage_context = StorageContext.from_defaults(
    persist_dir=INDEX_DIR,
    embed_model=embed_model
)
index = load_index_from_storage(storage_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para uma pergunta,
    usando o índice carregado.
    """
    q_engine = index.as_query_engine()
    result = q_engine.query(question)
    return str(result)
