import os
from llama_index import (
    StorageContext,
    load_index_from_storage,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "./storage"

# Cria context de serviço (com modelo de embedding)
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

# Se ainda não existe índice em disco, gera; senão, carrega
if not os.path.isdir(INDEX_DIR) or not os.listdir(INDEX_DIR):
    # 1) Leia o arquivo de transcrições
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # 2) Gere o índice
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    # 3) Persista em ./storage
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("✅ Índice criado e salvo em './storage'")
else:
    # Carrega índice existente
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)
    print("🔄 Índice carregado de './storage'")
