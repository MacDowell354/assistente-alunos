import os
import faiss
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente no local, se houver
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_URL = "storage/faiss.index"
EMB_URL   = "storage/embeddings.pkl"
TXT_FILE  = "transcricoes.txt"
CHUNK_SIZE = 1000

def chunk_text(text, size=CHUNK_SIZE):
    return [ text[i : i + size] for i in range(0, len(text), size) ]

def build_index():
    client = OpenAI(api_key=OPENAI_API_KEY)
    # 1) leia e parta o texto em chunks
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)

    # 2) calcule embeddings para cada chunk
    embeddings = []
    for chunk in chunks:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(resp.data[0].embedding)
    embeddings = np.array(embeddings).astype("float32")

    # 3) crie o índice FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 4) persista índice e chunks
    os.makedirs(os.path.dirname(INDEX_URL), exist_ok=True)
    faiss.write_index(index, INDEX_URL)
    with open(EMB_URL, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Índice FAISS gerado em '{INDEX_URL}', {len(chunks)} chunks.")

if __name__ == "__main__":
    build_index()
