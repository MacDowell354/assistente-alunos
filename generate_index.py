import os
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    GPTVectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPT_FILE = "transcricoes.txt"
INDEX_DIR = "storage"

def build_index():
    from llama_index.readers.file.base import SimpleFileReader

    print("Lendo transcricoes.txt...")
    text = open(TRANSCRIPT_FILE, "r", encoding="utf-8").read()
    docs = [SimpleFileReader(input_files=[]).load_data(text)]

    # embeddings & LLM
    embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, llm_predictor=LLMPredictor(llm=llm)
    )

    # cria índice
    print("Gerando índice vetorial...")
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    # salva em disco
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Índice salvo em ./{INDEX_DIR}")

if __name__ == "__main__":
    build_index()
