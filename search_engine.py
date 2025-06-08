import os
from llama_index import load_index_from_storage
from llama_index.storage import StorageContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
TOP_K = 3

# Carrega índice da pasta
storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_ctx)
query_engine = index.as_query_engine()

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna os TOP_K trechos mais relevantes para a pergunta.
    """
    resp = query_engine.query(question, similarity_top_k=TOP_K)
    # 'resp' já vem como objeto que imprime o texto concatenado
    return str(resp)
