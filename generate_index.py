import os

# leitor de diretório
from llama_index.readers.simple_directory_reader import SimpleDirectoryReader
# índice vetorial
from llama_index.indices.vector_store import GPTVectorStoreIndex
# contexto de storage
from llama_index.storage.storage_context import StorageContext
# embeddings OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# contexto de serviço (para configurar o embedding)
from llama_index.service_context import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# --- Prepara o ServiceContext com o modelo de embedding ---
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# --- Garante que ./storage exista ---
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Gera o índice apenas se não existir nada em storage ---
if not os.listdir(INDEX_DIR):
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    storage_context.persist(persist_dir=INDEX_DIR)

    print("✅ Índice gerado em", INDEX_DIR)
else:
    print("ℹ️  Índice já existe em", INDEX_DIR)
