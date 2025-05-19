import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
from llama_index.core import Settings

# === CONFIGURAÇÕES ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICAR A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === LER O ARQUIVO .TXT ===
if not os.path.exists(TRANSCRIPTION_FILE):
    raise FileNotFoundError(f"O arquivo {TRANSCRIPTION_FILE} não foi encontrado.")

with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# === DIVIDIR EM CHUNKS ===
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([SimpleDirectoryReader(input_files=[TRANSCRIPTION_FILE]).load_data()[0]])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# === CRIAR DOCUMENTOS E ÍNDICE ===
index = VectorStoreIndex(nodes)

# === SALVAR O ÍNDICE ===
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
