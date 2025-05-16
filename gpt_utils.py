import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(question, context, model="gpt-4"):
    prompt = f"""
    Você é um assistente inteligente para alunos. Responda à pergunta abaixo com base apenas no conteúdo fornecido no contexto. Se não encontrar a resposta no contexto, diga que não sabe.

    Contexto:
    {context}

    Pergunta: {question}
    Resposta:
    """
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Você é um assistente educacional útil."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message['content'].strip()
