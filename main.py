import os
import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Form, Depends, HTTPException, status, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- App & Templates ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- JWT / Auth Config ---
SECRET_KEY = os.getenv("SECRET_KEY", "troque_disso_no_env")
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str):
    if username not in fake_users or not verify_password(password, fake_users[username]):
        return False
    return username

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return username

@app.exception_handler(HTTPException)
async def auth_exception_handler(request, exc):
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
def login_post(request: Request, response: Response,
               username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Usuário ou senha inválidos."})
    access_token = create_access_token({"sub": user}, timedelta(minutes=ACCESS_EXPIRE_MINUTES))
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie("access_token", access_token, httponly=True, max_age=ACCESS_EXPIRE_MINUTES * 60)
    return response

@app.get("/chat", response_class=HTMLResponse)
def chat_get(request: Request, username: str = Depends(get_current_user)):
    # histórico vazio inicialmente
    history = []
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})

@app.post("/chat", response_class=HTMLResponse)
def chat_post(request: Request, question: str = Form(...), username: str = Depends(get_current_user)):
    # recupera histórico do form hidden
    data = await request.form()
    history = json.loads(data.get("history_json", "[]"))
    # insere pergunta nova
    history.append({"role": "user", "text": question})
    # busca contexto semântico
    context = retrieve_relevant_context(question)
    # gera resposta via OpenAI
    answer = generate_answer(question, context, history)
    history.append({"role": "assistant", "text": answer})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": history,
        "history_json": json.dumps(history)
    })
