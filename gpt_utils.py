import os
import openai
from typing import List, Dict
from search_engine import retrieve_relevant_context

# --- Configurações ---
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_HISTORY = 10  # quantas mensagens anteriores manter

# Prompt base do sistema
SYSTEM_PROMPT = """
Você é a IA.Nanda Mac, assistente virtual do curso Consultório High Ticket. 
Responda sempre de forma clara, objetiva e fundamentada no conteúdo do curso.
Se precisar usar exemplos, baseie-se no material de transcrições fornecido.
""".strip()


def build_chat_messages(
    history: List[Dict[str, str]],
    user_question: str
) -> List[Dict[str, str]]:
    """
    Constrói a lista de mensagens para enviar ao OpenAI, incluindo:
    - system prompt
    - contexto semântico (trecho mais relevante das transcrições)
    - histórico de conversa
    - a nova pergunta do usuário
    """
    messages: List[Dict[str, str]] = []
    # 1) mensagem do sistema
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2) contexto relevante extraído por busca semântica
    context = retrieve_relevant_context(user_question)
    if context:
        messages.append({
            "role": "system",
            "content": f"🚀 Contexto relevante das aulas:\n\n{context}"
        })

    # 3) últimas N mensagens do histórico (user + assistant)
    for msg in history[-MAX_HISTORY:]:
        messages.append(msg)

    # 4) pergunta atual do usuário
    messages.append({"role": "user", "content": user_question})

    return messages


def generate_answer(
    history: List[Dict[str, str]],
    user_question: str
) -> Dict[str, str]:
    """
    Recebe o histórico de conversa e a pergunta do usuário,
    retorna o par atualizado {role:content} e a resposta gerada.
    """
    # monta o prompt completo
    messages = build_chat_messages(history, user_question)

    # chama o OpenAI ChatCompletion
    completion = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages
    )

    answer = completion.choices[0].message.content.strip()

    # atualiza o histórico
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "history": history}
