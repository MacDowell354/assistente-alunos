import os
import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from passlib.context import CryptContext
from jose import JWTError, jwt

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- Configurações Auth ---
SECRET_KEY = os.getenv("SECRET_KEY", "mude_para_uma_chave_secreta_super_forte")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    # senha já será armazenada hash:
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> str | None:
    hashed = fake_users.get(username)
    if not hashed or not verify_password(password, hashed):
        return None
    # cria um token JWT
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": username, "exp": expire.isoformat()}
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

async def get_current_user(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

# --- App & Templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Intercepta 401 para redirecionar ao login
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        response = RedirectResponse(url="/login")
        # opcional: cookie de mensagem de erro
        response.set_cookie("login_error", "Sessão expirada – faça login novamente.", max_age=5)
        return response
    return HTMLResponse(str(exc.detail), status_code=exc.status_code)


# --- Rotas de Login ---
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    error = request.cookies.get("login_error")
    response = templates.TemplateResponse("login.html", {"request": request, "error": error})
    # limpa a mensagem
    response.delete_cookie("login_error")
    return response

@app.post("/login")
async def login_post(response: Response, username: str = Form(...), password: str = Form(...)):
    token = authenticate_user(username, password)
    if not token:
        # falha no login
        resp = templates.TemplateResponse(
            "login.html",
            {"request": Request({}), "error": "Usuário ou senha inválidos."},
        )
        return resp
    # sucesso: grava cookie e redireciona
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie("access_token", token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return response


# --- Chat Encadeado ---
@app.get("/", response_class=HTMLResponse)
async def chat_get(request: Request, user: str = Depends(get_current_user)):
    # histórico vazio para começar
    history: list[dict] = []
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "user": user, "history": history},
    )

@app.post("/", response_class=HTMLResponse)
async def chat_post(
    request: Request,
    question: str = Form(...),
    user: str = Depends(get_current_user),
):
    # lê histórico anterior do formulário (campo hidden JSON)
    form = await request.form()
    raw_history = form.get("history", "[]")
    history = json.loads(raw_history)

    # 1) Busca semântica no índice
    contexto = retrieve_relevant_context(question)

    # 2) Gera resposta juntando pergunta, contexto e histórico
    resposta = generate_answer(question, contexto, history)

    # 3) Atualiza histórico
    history.append({"role": "Você", "text": question})
    history.append({"role": "IA",    "text": resposta})

    # 4) Renderiza de volta
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": user,
            "history": history,
        },
    )
