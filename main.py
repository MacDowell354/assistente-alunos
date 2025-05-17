from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from responder import generate_answer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    context = ""  # Por enquanto, sem contexto
    answer = generate_answer(question, context)
    return templates.TemplateResponse("response.html", {
        "request": request,
        "question": question,
        "answer": answer
    })
