# gpt_utils.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(prompt: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
    )
    return resp.choices[0].message.content.strip()
