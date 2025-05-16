#import json
import os
import openai
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")
INDEX_PATH = "search_index_structured.json"

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_context(question, top_k=3):
    question_vector = get_embedding(question)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scored_chunks = [(cosine_similarity(question_vector, item["embedding"]), item["text"]) for item in data]
    scored_chunks.sort(reverse=True)
    top_chunks = [text for _, text in scored_chunks[:top_k]]
    return "\n\n".join(top_chunks)
