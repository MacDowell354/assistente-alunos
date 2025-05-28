import os
from datetime import datetime, timedelta

from fastapi import FastAPI, Form, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

from passlib.context import CryptContext
from jose import jwt, JWTError

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

# --- Configurações do App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Configurações de Autenticação JWT e OAuth2 ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Usuários em memória (exemplo); migrar para DB conforme evolução
type_ignored = None
fake_users = {
    "aluno1": pwd_ctx.hash("N4nd@M4c#2025")
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# --- Funções de segurança ---
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

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido ou expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # type: ignore
        if not username or username not in fake_users:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# --- Tratamento de erros de autenticação ---
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        if request.headers.get("accept", "").startswith("application/json"):
            return JSONResponse(status_code=401, content={"detail": exc.detail})
        return RedirectResponse(url="/login", status_code=303)
    raise exc

# --- Rotas de Login (OAuth2 Password) ---
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
    access_token = create_access_token(user)
    return {"access_token": access_token, "token_type": "bearer"}

# --- Rotas Protegidas ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, username: str = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(
    request: Request,
    question: str = Form(...),
    username: str = Depends(get_current_user),
):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    # JSON para AJAX
    if request.headers.get("accept", "").startswith("application/json"):
        return JSONResponse({"question": question, "answer": answer})
    # Fallback HTML
    return templates.TemplateResponse(
        "response.html",
        {"request": request, "question": question, "answer": answer},
    )
