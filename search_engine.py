import os
import pickle
import faiss
import numpy as np
import openai

# --- Configurações ---
openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
TOP_K = 3  # quantos chunks retornar

# 1) Carrega o índice FAISS e os chunks
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as fh:
    chunks = pickle.load(fh)

def retrieve_relevant_context(question: str) -> str:
    """
    Retorna os TOP_K chunks mais relevantes para 'question',
    concatenados como contexto para o prompt.
    """
    # 2) Embedding da pergunta
    resp = openai.embeddings.create(model="text-embedding-3-small", input=question)
    q_emb = np.array([resp["data"][0]["embedding"]], dtype="float32")

    # 3) Busca no FAISS
    _, I = index.search(q_emb, TOP_K)

    # 4) Seleciona e junta os chunks encontrados
    selected = [chunks[i] for i in I[0]]
    return "\n\n---\n\n".join(selected)
