import os
from llama_index.storage.storage_context import StorageContext
from llama_index import load_index_from_storage, GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext

# Configuração da sua chave e diretório de índice
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "./storage"

# 1) Gere as embeddings com sua chave
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

# 2) Leia transcricoes.txt e crie o índice (ou recarregue)
if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
    # primeira vez: gera índice
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("Índice criado e salvo em ./storage")
else:
    # já existe: apenas carrega
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    print("Índice carregado de ./storage")
