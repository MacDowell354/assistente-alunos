import os
import openai
from typing import List, Tuple
from search_engine import retrieve_relevant_context

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.2
MAX_TOKENS = 800

def generate_answer(
    question: str,
    chat_history: List[Tuple[str, str]]
) -> str:
    # 1) busca o contexto
    context = retrieve_relevant_context(question)

    # 2) mensagens montadas
    messages = [{
        "role": "system",
        "content": (
            "Você é um assistente do curso Consultório High Ticket. "
            "Use apenas este trecho das transcrições para responder:\n\n"
            f"{context}\n\n"
            "Se a pergunta for fora desse contexto, responda mantendo o foco no curso."
        )
    }]

    # 3) histórico
    for u, a in chat_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    # 4) pergunta atual
    messages.append({"role": "user", "content": question})

    # 5) envia à OpenAI
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()
