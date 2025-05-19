import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"

# === EMBEDDING SETTINGS ===
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# === LOAD INDEX ===
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

# === FASTAPI ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === RENDER HTML ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === RESPOSTA GPT ===
@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request):
    form = await request.form()
    question = form.get("question")

    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        answer = str(response)
    except Exception as e:
        answer = "Desculpe, houve um erro ao gerar a resposta."

    return templates.TemplateResponse(
        "response.html",
        {"request": request, "question": question, "answer": answer}
    )
