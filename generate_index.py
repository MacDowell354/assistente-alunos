# generate_index.py
import os
from llama_index import (
    StorageContext,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    Document,
    load_index_from_storage,
    GPTVectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.node_parser import SimpleNodeParser

# Caminhos / Configuração
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "storage"
TRANSCRIPT_FILE = "transcricoes.txt"

def build_and_persist_index():
    # Lê todo o arquivo de transcrições como um único Document
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    doc = Document(text)

    # Configura embedding e LLM
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm),
        embed_model=embed_model,
    )

    # Cria ou re-cria o índice
    index = GPTVectorStoreIndex.from_documents([doc], service_context=service_context)

    # Persiste em disco
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context.persist()

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    build_and_persist_index()
    print("✅ Índice gerado com sucesso em", INDEX_DIR)
