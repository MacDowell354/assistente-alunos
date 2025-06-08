import os

from llama_index import ServiceContext
from llama_index.readers import SimpleDirectoryReader
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Garante que a pasta existe
os.makedirs(INDEX_DIR, exist_ok=True)

# Se não existir índice, gera; senão, só informa
if not os.listdir(INDEX_DIR):
    # 1) Lê o TXT
    reader = SimpleDirectoryReader(input_files=["transcricoes.txt"])
    docs = reader.load_data()

    # 2) Configura o embedding
    embed = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    service_ctx = ServiceContext.from_defaults(embed_model=embed)

    # 3) Monta e persiste o índice
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_ctx)
    index.storage_context.persist(INDEX_DIR)
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
