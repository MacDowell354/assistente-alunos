import os
import openai
import numpy as np
import json
from tiktoken import get_encoding

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_PATH = "search_index_structured.json"

# Versão mínima de funcionamento enquanto o índice real não está presente
def retrieve_relevant_context(question, top_k=3):
    if not os.path.exists(INDEX_PATH):
        return ""  # Nenhum contexto ainda

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    question_embedding = get_embedding(question)
    scored_chunks = []

    for item in index_data:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored_chunks.append((score, item["text"]))

    scored_chunks.sort(reverse=True)
    top_chunks = [text for _, text in scored_chunks[:top_k]]
    return "\n\n".join(top_chunks)

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
