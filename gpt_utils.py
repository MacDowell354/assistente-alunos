import os
import openai
from typing import List, Tuple
from search_engine import retrieve_relevant_context

# --- Configurações da OpenAI ---
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.2
MAX_TOKENS = 800

def generate_answer(
    question: str,
    chat_history: List[Tuple[str, str]]
) -> str:
    """
    Gera uma resposta da OpenAI para a `question`, injetando
    primeiro o trecho mais relevante das transcrições e
    preservando o `chat_history` para encadeamento.
    """
    # 1) Busca o contexto relevante no índice
    context = retrieve_relevant_context(question)

    # 2) Monta a lista de mensagens para o ChatCompletion
    messages = []

    # Mensagem de sistema com o contexto
    messages.append({
        "role": "system",
        "content": (
            "Você é um assistente especializado no curso Consultório High Ticket. "
            "Use apenas o seguinte trecho das transcrições para responder:\n\n"
            f"{context}\n\n"
            "Se for pergunta fora desse contexto, responda com suas próprias palavras "
            "mas mantendo o foco no material do curso."
        )
    })

    # 3) Reproduz o histórico (usuário + assistente)
    for user_q, assistant_a in chat_history:
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a})

    # 4) Adiciona a pergunta atual
    messages.append({"role": "user", "content": question})

    # 5) Chama a OpenAI
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # 6) Extrai e devolve a resposta
    answer = resp.choices[0].message.content.strip()
    return answer
