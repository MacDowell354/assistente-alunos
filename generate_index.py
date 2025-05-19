import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core.storage import StorageContext
from llama_index.core.index import load_index_from_storage
from llama_index.core.service_context import ServiceContext

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === LER TRANSCRIÇÃO ===
print("📄 Lendo conteúdo do arquivo...")
with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# === DIVIDIR EM CHUNKS ===
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([Document(text=full_text)])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# === CRIAR E SALVAR O ÍNDICE ===
print("💾 Salvando índice em:", OUTPUT_DIR)
index = VectorStoreIndex(nodes, service_context=service_context)
storage_context = index.storage_context
storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
