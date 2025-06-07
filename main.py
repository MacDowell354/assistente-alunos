# src/main.py

import os
from datetime import timedelta, datetime

from fastapi import FastAPI, Form, Depends, HTTPException, status, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# FastAPI + static + templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# fake database
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

def verify_password(plain, hashed):
    return pwd_ctx.verify(plain, hashed)

def authenticate_user(username: str, password: str):
    if username in fake_users and verify_password(password, fake_users[username]):
        return username
    return None

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(lambda request: request.cookies.get("token"))):
    if not token:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except JWTError:
        raise HTTPException(status_code=401)

@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    # redireciona para login com flash
    resp = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    resp.set_cookie("token", "", max_age=0)
    resp.set_cookie("error", exc.detail, max_age=5)
    return resp

@app.get("/", response_class=HTMLResponse)
def root(req: Request):
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
def login_page(req: Request):
    error = req.cookies.get("error")
    resp = templates.TemplateResponse("login.html", {"request": req, "error": error})
    resp.delete_cookie("error")
    return resp

@app.post("/login")
def login(req: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos.")
    token = create_access_token({"sub": user})
    response = RedirectResponse(url="/chat", status_code=302)
    response.set_cookie(key="token", value=token, httponly=True)
    return response

@app.get("/chat", response_class=HTMLResponse)
def chat_page(req: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("chat.html", {"request": req, "history": []})

@app.post("/chat", response_class=HTMLResponse)
def chat(req: Request, question: str = Form(...), user: str = Depends(get_current_user), history: list = Form(default=[])):
    # recupera contexto semântico
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    # anexa à história de chat
    hist = history + [{"user": question, "assistant": answer}]
    return templates.TemplateResponse("chat.html", {"request": req, "history": hist})
