import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(question, context=""):
    prompt = f"{context}\n\nPergunta: {question}\nResposta:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente que responde perguntas com base no conteúdo do curso."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message["content"].strip()
