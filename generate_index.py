import os
from llama_index.readers.simple_directory_reader import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Cria o ServiceContext com o embedding da OpenAI
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# Garante que o diretório de índice exista
os.makedirs(INDEX_DIR, exist_ok=True)

# Se não encontrar nada em ./storage, gera; senão, passa adiante
if not os.listdir(INDEX_DIR):
    print("🗂️  Gerando índice em", INDEX_DIR)
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    # Persiste em disco
    storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_ctx
    storage_ctx.persist(persist_dir=INDEX_DIR)
    print("✅ Índice salvo em", INDEX_DIR)
else:
    print("ℹ️  Índice já existe em", INDEX_DIR)
