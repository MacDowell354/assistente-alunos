import os
from llama_index import Document, GPTVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# Cria pasta de índice se não existir
os.makedirs(INDEX_DIR, exist_ok=True)

# Configura o serviço de embeddings
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

# Se não houver nada em storage, gera o índice
if not os.listdir(INDEX_DIR):
    # Lê todo o arquivo de transcrições
    with open("transcricoes.txt", encoding="utf-8") as f:
        texto = f.read()

    # Cria um único Documento (você pode particionar em chunks se preferir)
    doc = Document(text=texto, doc_id="transcricoes")

    # Gera o índice semântico
    index = GPTVectorStoreIndex.from_documents([doc], service_context=service_context)

    # Salva no disco
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))

    print("Índice criado em", INDEX_DIR)
else:
    print("Índice já existe em", INDEX_DIR)
