from search_engine import retrieve_relevant_context
# ...
@app.post("/ask", response_class=HTMLResponse, dependencies=[Depends(get_current_user)])
async def ask(request: Request, question: str = Form(...)):
    # histórico já vindo de form-hidden ou sessão
    history = load_history(request)

    # busca semântica + resposta
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context, history)

    history.append({"role":"Você","text":question})
    history.append({"role":"IA","text":answer})

    return templates.TemplateResponse("chat.html", {
      "request": request,
      "history": history
    })
