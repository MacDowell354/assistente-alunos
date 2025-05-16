from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request, Form

from gpt_utils import generate_answer
from search_engine import retrieve_relevant_context

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, model="gpt-4")
    return templates.TemplateResponse("response.html", {
        "request": request,
        "question": question,
        "answer": answer,
        "context": context
    })
