# ... demais imports ...
from fastapi import FastAPI, Request, Depends, Response, status, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
# ...
from search_engine import retrieve_relevant_context
# ...

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# endpoint do chat, usando o retrieve_relevant_context
@app.post("/ask")
async def ask(request: Request, question: str = Form(...), token: str = Depends(get_current_user)):
    # monta o prompt com histórico...
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, history=...)
    return templates.TemplateResponse("chat.html", {"request": request, "history": updated_history})
