import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURAÇÕES ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICAR CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === DEFINIR EMBEDDING ===
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# === LER TRANSCRIÇÃO ===
print("📄 Lendo conteúdo do arquivo...")
with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# === DIVIDIR EM TRECHOS ===
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([SimpleDirectoryReader.input_to_doc(full_text)])

# === CRIAR ÍNDICE ===
print("🧠 Gerando embeddings e criando índice...")
index = VectorStoreIndex(nodes)

# === SALVAR O ÍNDICE ===
print(f"💾 Salvando índice em: {OUTPUT_DIR}")
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
