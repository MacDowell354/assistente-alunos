import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from openai import OpenAI

# CONFIG
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTIONS_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# Set API key and embedding model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# READ TRANSCRIPTIONS
print("📄 Lendo conteúdo do arquivo...")
if not os.path.exists(TRANSCRIPTIONS_FILE):
    raise FileNotFoundError(f"Arquivo {TRANSCRIPTIONS_FILE} não encontrado.")
with open(TRANSCRIPTIONS_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# PARSE INTO NODES
print("✂️ Dividindo em trechos...")
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(
    [SimpleDirectoryReader.input_to_doc(full_text)]
)

# GENERATE INDEX
print("🧠 Gerando embeddings com OpenAI...")
index = VectorStoreIndex(nodes)

# SAVE INDEX
print("💾 Salvando índice em:", OUTPUT_DIR)
storage_context = StorageContext.from_defaults()
index.storage_context.persist(persist_dir=OUTPUT_DIR)
print("✅ Índice gerado com sucesso!")
