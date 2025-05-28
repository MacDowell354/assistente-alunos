import os
from datetime import datetime, timedelta

from fastapi import FastAPI, Form, Depends, HTTPException, status, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm

from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- Configurações do App ---
app = FastAPI()

# Diretório de estáticos e templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Configurações de Autenticação JWT ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Exemplo de usuário fixo; depois pode migrar para um DB real
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

# Redireciona pro /login em caso de 401
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return RedirectResponse(url="/login")
    raise exc

# --- Rotas de Login ---
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    token = create_access_token(user)
    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    resp.set_cookie(
        "access_token",
        token,
        httponly=True,
        max_age=ACCESS_EXPIRE * 60
    )
    return resp

# --- Rotas Protegidas ---
@app.get("/", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def ask_question(request: Request, question: str = Form(...)):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    return templates.TemplateResponse(
        "response.html",
        {
            "request": request,
            "question": question,
            "answer": answer
        }
    )

