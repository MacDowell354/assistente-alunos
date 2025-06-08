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

# --- Configurações do Auth ---
SECRET_KEY = os.getenv("SECRET_KEY", "mude_para_uma_chave_forte_aqui")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash(os.getenv("DEFAULT_STUDENT_PWD", "N4nd@M4c#2025"))
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> str | None:
    if username in fake_users and verify_password(password, fake_users[username]):
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
        return token
    return None

async def get_current_user(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


# --- Instancia o FastAPI e templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Captura 401 e redireciona p/ login
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        resp = RedirectResponse(url="/login", status_code=302)
        resp.set_cookie("login_error", "Sessão expirada. Faça login novamente.", max_age=5)
        return resp
    return HTMLResponse(str(exc.detail), status_code=exc.status_code)


# --- Rotas de Login ---
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    error = request.cookies.get("login_error")
    resp = templates.TemplateResponse("login.html", {"request": request, "error": error})
    resp.delete_cookie("login_error")
    return resp

@app.post("/login")
async def login_post(
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    token = authenticate_user(username, password)
    if not token:
        return templates.TemplateResponse(
            "login.html",
            {"request": Request({}), "error": "Usuário ou senha inválidos."},
            status_code=401
        )
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES*60)
    return resp


# --- Chat Encadeado ---
@app.get("/", response_class=HTMLResponse)
async def chat_get(request: Request, user: str = Depends(get_current_user)):
    history: list[dict] = []
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})

@app.post("/", response_class=HTMLResponse)
async def chat_post(
    request: Request,
    question: str = Form(...),
    user: str = Depends(get_current_user)
):
    form = await request.form()
    raw_history = form.get("history", "[]")
    history = json.loads(raw_history)

    # 1) busca semântica
    context = retrieve_relevant_context(question)

    # 2) gera resposta com prompt + contexto
    answer = generate_answer(question, context, history)

    # 3) atualiza histórico
    history.append({"role": "Você", "text": question})
    history.append({"role": "IA",   "text": answer})

    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "history": history}
    )
