# main.py
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Form, Request, Response, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from jose import JWTError, jwt

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# App e templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Auth / JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def verify_password(plain, hashed) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str):
    hashed = fake_users.get(username)
    if not hashed or not verify_password(password, hashed):
        return False
    return {"sub": username}

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise JWTError("Sessão inválida")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload.get("sub")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        user = await get_current_user(request)
        return templates.TemplateResponse("chat.html", {"request": request, "history": []})
    except JWTError:
        return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login_post(response: Response, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        return RedirectResponse(url="/login?error=Usuário%20ou%20senha%20inválidos", status_code=302)
    token = create_access_token({"sub": username}, timedelta(minutes=ACCESS_EXPIRE_MINUTES))
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie("access_token", token, httponly=True, max_age=ACCESS_EXPIRE_MINUTES * 60)
    return response

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    # Recupera histórico enviado no form (parte encoded via JS)
    form = await request.form()
    history = form.get("history", "")
    history_pairs = []
    if history:
        for item in history.split("||"):
            user_q, ia_a = item.split("::")
            history_pairs.append((user_q, ia_a))

    # Busca contexto na transcrição
    context = retrieve_relevant_context(question)

    # Monta prompt encadeado
    messages = []
    for uq, aa in history_pairs:
        messages.append({"role": "user", "content": uq})
        messages.append({"role": "assistant", "content": aa})
    # Insere contexto como system
    messages.insert(0, {"role": "system", "content": f"Use este contexto do curso:\n{context}"})
    messages.append({"role": "user", "content": question})

    # Gera resposta final
    answer = generate_answer(messages)

    # Atualiza histórico e renderiza
    history_pairs.append((question, answer))
    encoded = "||".join(f"{uq}::{aa}" for uq, aa in history_pairs)
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "history": history_pairs, "access": str(encoded)}
    )
