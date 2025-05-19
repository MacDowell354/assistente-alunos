import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.readers.file.docs import PDFReader
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import get_nodes_from_documents
from openai import OpenAI

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# ===== Setup =====
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# ===== Load Data =====
print("📄 Lendo conteúdo do arquivo...")
with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

print("✂️ Dividindo em trechos...")
parser = MarkdownElementNodeParser()
documents = [SimpleDirectoryReader.input_to_doc(full_text)]
nodes = get_nodes_from_documents(documents, parser)

# ===== Index & Save =====
print("🧠 Gerando embeddings com OpenAI...")
index = VectorStoreIndex(nodes)
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
