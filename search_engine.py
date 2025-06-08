import os

from llama_index import load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# 1) Mesma embedding usada no build
embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
service_ctx = ServiceContext.from_defaults(embed_model=embed_model)

# 2) Carrega índice FAISS
storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_ctx, service_context=service_ctx)

def retrieve_relevant_context(question: str) -> str:
    """
    Busca os TOP-3 documentos mais relevantes e retorna a concatenação deles.
    """
    qe = index.as_query_engine(similarity_top_k=3)
    resp = qe.query(question)
    return str(resp)
