import os

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa o modelo de embedding com a sua chave
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# Carrega o índice gerado em disk
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna o contexto mais relevante para a pergunta,
    usando o índice carregado em memória.
    """
    # Cria um query engine e obtém a resposta
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    # Converte para string e devolve
    return str(response)
#Atualiza search_engine.py e adiciona retrieve_relevant_context
