import os

from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
TRANSCRIPTS_FILE = "transcricoes.txt"

def build_index():
    # 1) Inicializa embedding model
    embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    service_ctx = ServiceContext.from_defaults(embed_model=embed_model)

    # 2) Lê as transcrições
    docs = SimpleDirectoryReader(input_files=[TRANSCRIPTS_FILE]).load_data()

    # 3) Cria índice FAISS
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_ctx)

    # 4) Persiste em disco
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"✅ Índice FAISS criado em '{INDEX_DIR}' com {len(docs)} documentos.")

if __name__ == "__main__":
    build_index()
