import os
import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- Auth Config ---
SECRET_KEY = os.getenv("SECRET_KEY", "troque_para_uma_chave_super_secreta")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DEFAULT_PWD = os.getenv("DEFAULT_STUDENT_PWD", "N4nd@M4c#2025")

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {"aluno1": pwd_ctx.hash(DEFAULT_PWD)}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> str | None:
    hashed = fake_users.get(username)
    if not hashed or not verify_password(password, hashed):
        return None
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
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

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    error = request.cookies.get("login_error")
    resp = templates.TemplateResponse("login.html", {"request": request, "error": error})
    resp.delete_cookie("login_error")
    return resp

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, response: Response,
                     username: str = Form(...), password: str = Form(...)):
    token = authenticate_user(username, password)
    if not token:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Usuário ou senha inválidos."},
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    resp = RedirectResponse(url="/chat", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES*60)
    return resp

@app.get("/chat", response_class=HTMLResponse)
async def chat_get(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/ask", response_class=HTMLResponse)
async def chat_post(request: Request, question: str = Form(...), user: str = Depends(get_current_user)):
    form = await request.form()
    history = json.loads(form.get("history", "[]"))

    # busca contexto e gera resposta
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, history)

    history.append({"role": "Você", "text": question})
    history.append({"role": "IA",    "text": answer})

    return templates.TemplateResponse("chat.html", {"request": request, "history": history})

@app.get("/health")
async def health():
    return {"status": "ok"}
