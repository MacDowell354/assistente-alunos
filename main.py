import os
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
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users = {
    "aluno1": pwd_ctx.hash("senha123")
}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> str | None:
    if username in fake_users and verify_password(password, fake_users[username]):
        return username
    return None


def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE)
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # type: ignore
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if username not in fake_users:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return username


# --- Exception Handler: 401 → /login + flash message ---
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        resp = RedirectResponse(url="/login")
        # use hífen (-), não “—”
        resp.set_cookie(
            key="login_msg",
            value="Sessão expirada - faça login novamente.",
            max_age=5,        # dura 5s
            httponly=True
        )
        return resp
    raise exc


# --- Login Routes ---
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    # limpa o flash-message
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.delete_cookie("login_msg")
    return resp


@app.post("/login")
async def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...)
):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Usuário ou senha inválidos."},
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    token = create_access_token(user)
    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    resp.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=ACCESS_EXPIRE * 60
    )
    return resp


# --- Protected Routes ---
@app.get("/", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def ask_question(request: Request, question: str = Form(...)):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    return templates.TemplateResponse(
        "response.html",
        {"request": request, "question": question, "answer": answer}
    )
