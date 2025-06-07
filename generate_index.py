import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Inicializa o embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Cria pasta de índice, se ainda não existir
os.makedirs(INDEX_DIR, exist_ok=True)

# Se não houver arquivo de índice, gera um
index_path = os.path.join(INDEX_DIR, "index.json")
if not os.path.exists(index_path):
    # Carrega todas as transcrições
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()
    # Cria o índice e persiste em disco
    index = GPTVectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(
            persist_dir=INDEX_DIR,
            embed_model=embed_model
        )
    )
    index.storage_context.persist()
    print("Índice gerado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
