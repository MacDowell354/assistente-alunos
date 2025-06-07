# generate_index.py
import os
from llama_index import (
    StorageContext,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
)

# Lê o arquivo de transcrições
docs = SimpleDirectoryReader(input_files=["transcricoes.txt"]).load_data()

# Gera o índice
index = GPTVectorStoreIndex.from_documents(docs)

# Persistência em ./storage
storage_context = StorageContext.from_defaults(persist_dir="storage")
index.storage_context.persist()
print("Índice gerado em ./storage")
