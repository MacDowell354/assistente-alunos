from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context
from auth import get_current_user  # sua lógica de auth existente

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def chat_get(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/", response_class=HTMLResponse)
async def chat_post(request: Request, question: str = Form(...), user: str = Depends(get_current_user)):
    form = await request.form()
    history = json.loads(form.get("history", "[]"))

    # Busca semântica + geração de resposta
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, history)

    history.append({"role": "Você", "text": question})
    history.append({"role": "IA",    "text": answer})

    return templates.TemplateResponse("chat.html", {"request": request, "history": history})
