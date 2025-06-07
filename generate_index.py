# src/generate_index.py

import os
from llama_index.readers.simple_directory_reader import SimpleDirectoryReader
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# inicializa modelo de embedding
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# garante que a pasta existe
os.makedirs(INDEX_DIR, exist_ok=True)

if not os.listdir(INDEX_DIR):
    print("Gerando índice a partir de transcricoes.txt …")
    # lê todas as transcrições num único documento
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # monta um índice vetorial
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    # persiste
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))
    print("✅ Índice gerado em", INDEX_DIR)
else:
    print("⏭️ Índice já existe em", INDEX_DIR)
