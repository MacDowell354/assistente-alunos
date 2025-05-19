import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(question, context, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente útil e responde com base no conteúdo fornecido.",
            },
            {
                "role": "user",
                "content": f"{context}\n\n{question}",
            },
        ]
    )
    return response.choices[0].message.content
