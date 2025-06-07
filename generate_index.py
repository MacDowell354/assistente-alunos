import os
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    ServiceContext,
    OpenAIEmbedding,
)

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# --- Inicializa o ServiceContext com seu modelo de embedding ---
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
)

# --- Garante que a pasta de índice exista ---
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Se estiver vazia, gera o índice; senão só informa ---
if not os.listdir(INDEX_DIR):
    print("🗂️  Gerando índice pela primeira vez…")
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(
        docs,
        service_context=service_context,
    )
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    storage_context.persist(persist_dir=INDEX_DIR)
    print(f"✅ Índice gerado em {INDEX_DIR}")
else:
    print(f"ℹ️ Índice já existe em {INDEX_DIR}")
