import os
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    OpenAIEmbedding,
    ServiceContext,
)
from llama_index import LLMPredictor, PromptHelper

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# configura o embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# garante que o diretório existe
os.makedirs(INDEX_DIR, exist_ok=True)

# se ainda não houver arquivo de índice, cria um
if not os.listdir(INDEX_DIR):
    # lê todas as transcrições
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # gera o índice
    index = GPTVectorStoreIndex.from_documents(
        docs,
        service_context=service_context
    )
    # persiste em disco
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
