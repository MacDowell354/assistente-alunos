# src/gpt_utils.py

from datetime import datetime
import os
from openai import OpenAI
# ou: import openai

def generate_answer(question: str, context: str) -> str:
    """
    Aqui você monta o prompt com context + question
    e faz a chamada ao OpenAI completions/chat completions.
    """
    prompt = f"""
Você é um tutor no Curso “Consultório High Ticket”. Use apenas o contexto abaixo para responder:

CONTEXTOS:
{context}

ALUNO PERGUNTA:
{question}

Responda de forma clara, objetiva, como se fosse um mentor 1-a-1.
"""
    resp = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Você é um tutor."},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
