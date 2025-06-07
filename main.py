from fastapi import FastAPI, Form, Depends, Request, Response, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from responder import auth_required, auth_exception_handler  # Seu middleware de auth
from gpt_utils import generate_answer  # Geração de prompt + openai
from search_engine import retrieve_relevant_context

app = FastAPI(exception_handlers={Exception: auth_exception_handler})
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/ask", response_class=HTMLResponse)
@auth_required
async def ask(request: Request, question: str = Form(...), history: str = Form("")):
    # aqui você injeta o contexto semântico:
    context = retrieve_relevant_context(question)
    answer, new_history = generate_answer(question, context, history)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": new_history,
        "user_prompt": question,
        "ia_response": answer
    })
# ... resto das rotas (login, logout, index, etc.) ...
