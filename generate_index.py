import os, pickle, numpy as np, faiss, openai

# --- Configurações ---
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
CHUNK_SIZE = 1000

def build_index():
    # 1) Lê e chunkiza
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # 2) Gera embeddings com a nova API v1
    embs = []
    for chunk in chunks:
        resp = openai.Embeddings.create(model="text-embedding-3-small", input=chunk)
        embs.append(resp["data"][0]["embedding"])
    arr = np.array(embs, dtype="float32")

    # 3) Cria o índice FAISS
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    # 4) Persiste tudo
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)

    print("✅ Índice FAISS gerado com", len(chunks), "chunks.")

if __name__ == "__main__":
    # só gera se não existir
    if not os.path.exists(INDEX_PATH):
        build_index()
    else:
        print("ℹ️  Índice já existe.")
