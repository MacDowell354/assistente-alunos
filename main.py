import os
from datetime import datetime, timedelta

from fastapi import (
    FastAPI, Form, Depends,
    HTTPException, status,
    Request
)
from fastapi.responses import (
    HTMLResponse, RedirectResponse
)
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
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
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
        resp.set_cookie("login_msg", "Sessão expirada – faça login novamente.", max_age=5)
        return resp
    raise exc

# --- Login Routes ---
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.delete_cookie("login_msg")
    return resp

@app.post("/login")
async def login(
    request: Request,
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
    resp.set_cookie("access_token", token, httponly=True, max_age=ACCESS_EXPIRE*60)
    return resp

# --- Protected Chat Route ---
@app.get("/", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def chat_form(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def ask(request: Request, question: str = Form(...)):
    # Busca contexto
    context = retrieve_relevant_context(question)
    # Gera resposta considerando histórico + contexto
    answer = generate_answer(question, context)
    # Mantém histórico na sessão (via cookie ou hidden input, mas aqui simplificado)
    # Para persistência real, usar banco ou sessão server-side.
    history = request.cookies.get("chat_history", "")
    # Append novo turno
    history += f"Você: {question}\nIA: {answer}\n---\n"
    resp = templates.TemplateResponse(
        "chat.html",
        {"request": request, "history": history.split("\n---\n")},
    )
    resp.set_cookie("chat_history", history, httponly=True)
    return resp
