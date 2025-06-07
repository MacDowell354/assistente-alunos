import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.service_context import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# --- Inicializa o modelo de embedding via ServiceContext ---
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# --- Garante que a pasta ./storage exista ---
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Se não houver índice, gera; senão, não faz nada ---
if not os.listdir(INDEX_DIR):
    # Lê todas as transcrições
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # Cria o índice vetorial
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    # Persiste em disco
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
