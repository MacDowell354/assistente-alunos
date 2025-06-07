import os

from llama_index.core.storage_context import StorageContext
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, load_index_from_storage
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

# Cria o diretório se não existir
os.makedirs(INDEX_DIR, exist_ok=True)

# Gera ou carrega o índice
if not os.listdir(INDEX_DIR):
    # Lê todo o texto da transcricoes.txt
    documents = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
