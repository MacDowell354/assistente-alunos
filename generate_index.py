import os, pickle, numpy as np, faiss, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
CHUNK_SIZE = 1000

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    text = open("transcricoes.txt", encoding="utf-8").read()
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    embs = []
    for chunk in chunks:
        resp = openai.embeddings.create(model="text-embedding-3-small", input=chunk)
        embs.append(resp.data[0].embedding)
    arr = np.array(embs, dtype="float32")

    idx = faiss.IndexFlatL2(arr.shape[1])
    idx.add(arr)
    faiss.write_index(idx, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Índice FAISS criado com", len(chunks), "chunks.")

if __name__ == "__main__":
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        build_index()
    else:
        print("ℹ️  Índice já existe.")
