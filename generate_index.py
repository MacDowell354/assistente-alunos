import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.storage import StorageContext

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

def main():
    # Garante pasta de persistência
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1) Carrega o texto inteiro e faz o split interno em chunks
    docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()

    # 2) Cria índice FAISS
    index = GPTVectorStoreIndex.from_documents(docs)

    # 3) Persiste em disco
    storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_ctx
    index.save_to_disk(os.path.join(INDEX_DIR, "index.json"))

    print(f"✅ Indice FAISS criado em '{INDEX_DIR}' com {len(docs)} documentos.")

if __name__ == "__main__":
    main()
