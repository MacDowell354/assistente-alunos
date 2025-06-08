# generate_index.py
import os, pickle
import numpy as np
import faiss
import openai

# Configurações
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH      = os.path.join(INDEX_DIR, "chunks.pkl")
CHUNK_SIZE = 1000  # caracteres por chunk

def build_index():
    # 1) Lê todo o texto
    with open("transcricoes.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Divide em chunks
    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # 3) Gera embeddings via OpenAI
    embeddings = []
    for chunk in chunks:
        resp = openai.Embedding.create(input=[chunk], model="text-embedding-3-small")
        embeddings.append(resp["data"][0]["embedding"])
    arr = np.array(embeddings, dtype="float32")

    # 4) Cria índice FAISS e adiciona vetores
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)

    # 5) Persiste em disco
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"✅ Í­ndice FAISS gerado em '{INDEX_DIR}' com {len(chunks)} chunks.")

if __name__ == "__main__":
    # Só gera se não existir já
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        build_index()
    else:
        print("ℹ️  Índice já existe em storage.")
