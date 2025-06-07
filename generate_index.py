import os

from llama_index import (
    StorageContext,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa o embedding
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Garante que o diretório de índice exista
os.makedirs(INDEX_DIR, exist_ok=True)

# Se não existir índice, gera; senão, só informa
if not os.listdir(INDEX_DIR):
    # Lê o arquivo de transcrições
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # Cria índice vetorial
    index = GPTVectorStoreIndex.from_documents(docs)
    # Associa ao storage e salva em disco
    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
