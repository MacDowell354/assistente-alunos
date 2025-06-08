import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# inicializa o embedding
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# garante que o diretório de índice exista
os.makedirs(INDEX_DIR, exist_ok=True)

# se não existir índice, gera; senão, só informa
if not os.listdir(INDEX_DIR):
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    service_ctx = ServiceContext.from_defaults()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_ctx)
    storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.set_index_id("main")
    index.storage_context = storage_ctx
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
