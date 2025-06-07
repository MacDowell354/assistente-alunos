import os
from llama_index import (
    StorageContext,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import LLMPredictor, ServiceContext, PromptHelper

# --- Configurações de Embedding / Modelo ---
from llama_index.llms import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)
llm_predictor = LLMPredictor(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API_KEY)
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embed_model,
    prompt_helper=PromptHelper.from_llm_predictor(llm_predictor),
)

# --- Leitura das transcrições e criação do índice ---
if __name__ == "__main__":
    # cria pasta storage se não existir
    os.makedirs("storage", exist_ok=True)

    # lê o arquivo único de transcricoes.txt
    reader = SimpleDirectoryReader(input_files=["transcricoes.txt"])
    docs = reader.load_data()

    # gera o índice e salva em disco
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir="storage")

    print("Índice gerado e salvo em ./storage")
