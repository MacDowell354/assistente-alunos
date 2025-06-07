import os
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import LLMPredictor, ServiceContext
from openai import OpenAI

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Gera embeddings
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Lê as transcrições
docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()

# Cria contexto de serviço
service_context = ServiceContext.from_defaults(
    chunk_size_limit=1024,
    embed_model=embed_model,
)

# Cria e persiste o índice
index = VectorStoreIndex.from_documents(
    docs, service_context=service_context
)
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index.storage_context.persist(persist_dir=INDEX_DIR)

print("✅ Índice gerado com sucesso em", INDEX_DIR)
