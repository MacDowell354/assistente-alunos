import os, pickle, numpy as np, faiss, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
TOP_K = 3

idx = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

def retrieve_relevant_context(question: str) -> str:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=question)
    q_emb = np.array([resp.data[0].embedding], dtype="float32")
    _, I = idx.search(q_emb, TOP_K)
    return "\n\n---\n\n".join(chunks[i] for i in I[0])
