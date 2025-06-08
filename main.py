# No topo do arquivo (imports)
import os
from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# ... seus outros imports (jwt, passlib, etc)

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context  # <-- novo import

# --- App & Templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Dentro do endpoint de chat ---
@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...), token: str = Depends(get_current_user)):
    # 1) Recupera histórico (lista de Q/As já em session ou cookie)
    history = get_history_from_request(request)

    # 2) Busca contexto semântico
    context = retrieve_relevant_context(question)

    # 3) Chama seu helper para gerar a resposta do GPT,
    #    passando o contexto + histórico + pergunta nova
    answer = generate_answer(question, context=context, history=history)

    # 4) Renderiza o chat.html com todo histórico + nova Q/A
    new_history = history + [{"user": question, "ai": answer}]
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "history": new_history
        }
    )
