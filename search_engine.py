import os
import json
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.core.settings import Settings

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("⚠️ A variável de ambiente OPENAI_API_KEY não está definida.")

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
nodes = parser.get_nodes_from_documents([Document(text=full_text)])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# === CRIAR DOCUMENTOS E ÍNDICE ===
documents = [Document(text=node.text) for node in nodes]
index = VectorStoreIndex.from_documents(documents)

# === SALVAR O ÍNDICE DO LLAMAINDEX ===
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

# === SALVAR VERSÃO search_index_structured.json ===
print("📝 Salvando índice estruturado para uso no search_engine.py...")
index_data = []

for node in nodes:
    index_data.append({
        "text": node.text,
        "embedding": node.embedding
    })

with open("search_index_structured.json", "w", encoding="utf-8") as f:
    json.dump(index_data, f, ensure_ascii=False, indent=2)

print("✅ search_index_structured.json salvo com sucesso.")
print("🎉 Índice gerado com sucesso!")
