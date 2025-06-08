import os, pickle, faiss, numpy as np, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
TOP_K = 3

# 1) Carrega índice e chunks
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as fh:
    chunks = pickle.load(fh)

def retrieve_relevant_context(question: str) -> str:
    # 2) Embed da pergunta
    resp = openai.Embeddings.create(model="text-embedding-3-small", input=question)
    q_emb = np.array([resp["data"][0]["embedding"]], dtype="float32")

    # 3) Busca no FAISS
    _, I = index.search(q_emb, TOP_K)
    selected = [chunks[i] for i in I[0]]
    return "\n\n---\n\n".join(selected)
