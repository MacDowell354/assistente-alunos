from search_engine import retrieve_relevant_context

# dentro do seu endpoint de perguntas:
@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def ask(request: Request, question: str = Form(...)):
    history = load_history(request)  # sua função de recuperar histórico
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, history)
    history.append({"role": "Você", "text": question})
    history.append({"role": "IA",    "text": answer})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": history
    })
