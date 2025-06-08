# search_engine.py
import os, pickle
import faiss
import numpy as np
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH      = os.path.join(INDEX_DIR, "chunks.pkl")
TOP_K = 3  # quantos resultados retornar

# 1) Carrega índice e chunks
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNKS_PATH, "rb") as fh:
    chunks = pickle.load(fh)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna os TOP_K pedaços mais relevantes como contexto.
    """
    # 2) Embedding da pergunta
    resp = openai.Embedding.create(input=[question], model="text-embedding-3-small")
    q_emb = np.array([resp["data"][0]["embedding"]], dtype="float32")

    # 3) Busca FAISS
    D, I = index.search(q_emb, TOP_K)

    # 4) Concatena os chunks encontrados
    selected = [chunks[i] for i in I[0]]
    return "\n\n---\n\n".join(selected)
