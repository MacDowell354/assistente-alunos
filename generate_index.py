import os

# Leitor de arquivos locais
from llama_index.readers.simple_directory_reader import SimpleDirectoryReader
# Criação de índice vetorial
from llama_index.indices.vector_store import GPTVectorStoreIndex
# Persistência do índice
from llama_index.storage.storage_context import StorageContext
# Embeddings OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# Contexto de serviço (combina LLM + Embedding)
from llama_index.service_context import ServiceContext

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

# Garante que ./storage exista
os.makedirs(INDEX_DIR, exist_ok=True)

# Se estiver vazio, gera o índice; senão, apenas informa
if not os.listdir(INDEX_DIR):
    print("🗂️ Gerando índice a partir de transcricoes.txt…")
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_ctx
    storage_ctx.persist(persist_dir=INDEX_DIR)
    print(f"✅ Índice gerado em '{INDEX_DIR}'")
else:
    print(f"ℹ️  Índice já existe em '{INDEX_DIR}'")
