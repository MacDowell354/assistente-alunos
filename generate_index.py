import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from tqdm import tqdm

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === FUNÇÃO PARA LER O .TXT ===
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# === LER A TRANSCRIÇÃO ===
print("📄 Lendo conteúdo do arquivo...")
full_text = read_txt_file(TRANSCRIPTION_FILE)

# === DIVIDIR EM CHUNKS ===
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([TextNode(text=full_text)])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings e criando índice...")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# === CRIAR E SALVAR O ÍNDICE ===
index = VectorStoreIndex(nodes)
storage_context = index.storage_context
storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
