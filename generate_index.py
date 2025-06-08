import os
import pickle
import numpy as np
import faiss
import openai

# --- Configurações ---
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
CHUNK_SIZE = 1000  # caracteres por chunk

def build_index():
    # 1) Lê e chunkiza o texto
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # 2) Gera embeddings com a nova API openai.embeddings.create
    embeddings = []
    for chunk in chunks:
        resp = openai.embeddings.create(model="text-embedding-3-small", input=chunk)
        # acessa o embedding via resp.data[0].embedding
        embeddings.append(resp.data[0].embedding)
    arr = np.array(embeddings, dtype="float32")

    # 3) Cria e popula o índice FAISS
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    # 4) Persiste em disco
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"✅ Índice FAISS gerado em '{INDEX_DIR}' com {len(chunks)} chunks.")

if __name__ == "__main__":
    # Gera o índice apenas se não existir
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        build_index()
    else:
        print("ℹ️  Índice já existe em storage/")
