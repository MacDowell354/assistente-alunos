import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_URL = "storage/faiss.index"
EMB_URL   = "storage/embeddings.pkl"
TOP_K     = 3

# 1) inicializa cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 2) carrega índice e chunks
index = faiss.read_index(INDEX_URL)
with open(EMB_URL, "rb") as f:
    chunks = pickle.load(f)

def retrieve_relevant_context(question: str) -> str:
    # a) embedding da pergunta
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    q_emb = np.array(resp.data[0].embedding).astype("float32")[None, :]

    # b) busca TOP_K vizinhos
    D, I = index.search(q_emb, TOP_K)

    # c) retorna os chunks correspondentes
    selected = [ chunks[i] for i in I[0] ]
    return "\n\n---\n\n".join(selected)
