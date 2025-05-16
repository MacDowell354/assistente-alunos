import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente útil e responde com base no conteúdo fornecido."},
            {"role": "user", "content": f"{context}\n\n{question}"}
        ]
    )
    return response.choices[0].message["content"]
