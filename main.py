import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Form, Request, Response, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- App & Templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Auth Config ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str):
    if username in fake_users and verify_password(password, fake_users[username]):
        return username
    return None

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if user is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    return user

# --- Rotas ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Usuário ou senha inválidos."})
    token = create_access_token({"sub": user})
    resp = RedirectResponse(url="/chat", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, max_age=ACCESS_EXPIRE_MINUTES*60)
    return resp

@app.get("/chat", response_class=HTMLResponse)
async def chat_get(request: Request, user: str = Depends(get_current_user)):
    # histórico vazio no primeiro acesso
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/ask", response_class=HTMLResponse)
async def chat_ask(request: Request, question: str = Form(...), user: str = Depends(get_current_user)):
    # lê histórico da sessão (se quiser salvar em cookie/session)
    form = await request.form()
    history = json.loads(form.get("history", "[]"))

    # recupera contexto semântico da transcrição
    context = retrieve_relevant_context(question)

    # gera resposta passando contexto + histórico
    answer = await generate_answer(question, history=history, context=context)

    # atualiza histórico
    history.append({"role": "user", "text": question})
    history.append({"role": "assistant", "text": answer})

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": history
    })
