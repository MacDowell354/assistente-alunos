import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.schema import Document
from openai import OpenAI

# ==== CONFIG ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# ==== VERIFY KEY ====
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# ==== PREPARE DOCUMENT ====
if not os.path.exists(TRANSCRIPTION_FILE):
    raise FileNotFoundError(f"O arquivo {TRANSCRIPTION_FILE} não foi encontrado.")

# Copia o .txt para uma pasta temporária
os.makedirs("tmp", exist_ok=True)
os.system(f"cp {TRANSCRIPTION_FILE} tmp/conteudo.txt")

# ==== CRIAR ÍNDICE ====
print("📄 Lendo conteúdo do arquivo...")
with open(f"tmp/conteudo.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# ==== DIVIDIR EM CHUNKS ====
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
document = Document(text=full_text)
nodes = parser.get_nodes_from_documents([document])

# ==== EMBEDDINGS ====
print("🧠 Gerando embeddings com OpenAI...")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# ==== CRIAR DOCUMENTOS E ÍNDICE ====
index = VectorStoreIndex.from_documents(nodes)

# ==== SALVAR O ÍNDICE ====
print(f"💾 Salvando índice em: {OUTPUT_DIR}")
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
