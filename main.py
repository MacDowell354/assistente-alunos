import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

from search_engine import retrieve_relevant_context
from gpt_utils import generate_answer  # seu helper para chamar o GPT com contexto e histórico

# Carrega .env
load_dotenv()

# --- Config Auth JWT ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def authenticate_user(username: str, password: str):
    if username not in fake_users:
        return False
    return pwd_ctx.verify(password, fake_users[username])

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(lambda request: request.cookies.get("token"))):
    if not token:
        raise HTTPException(status_code=status.HTTP_302_FOUND, headers={"Location": "/login"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if user is None:
            raise
    except JWTError:
        raise HTTPException(status_code=status.HTTP_302_FOUND, headers={"Location": "/login"})
    return user

# --- FastAPI & Templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Redireciona "/" para /login
@app.get("/")
def root():
    return RedirectResponse(url="/login")

# Página de login
@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

# Processa login
@app.post("/login")
def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if not authenticate_user(username, password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Usuário ou senha inválidos."})
    token = create_access_token({"sub": username})
    response = RedirectResponse(url="/chat", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="token", value=token, httponly=True)
    return response

# Tela de chat
@app.get("/chat", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
def chat_get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

# Endpoint de perguntas
@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
def ask(request: Request, question: str = Form(...)):
    # recupere o histórico anterior do formulário (via campo hidden em chat.html)
    history = request.form().get("history", "[]")
    history = eval(history)  # simplificação; ideal usar json.loads

    # busca semântica + contexto
    context = retrieve_relevant_context(question)

    # gera resposta combinando histórico + contexto
    answer = generate_answer(question, context=context, history=history)

    new_history = history + [{"user": question, "ai": answer}]
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": new_history
    })
