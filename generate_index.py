import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.index_store import SimpleIndexStore

# ==== CONFIGURAÇÃO ====
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TXT_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# ==== VERIFICA A CHAVE ====
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# ==== LER A TRANSCRIÇÃO ====
print("📄 Lendo conteúdo do arquivo...")
with open(TXT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# ==== DIVIDIR EM CHUNKS ====
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.split_text(full_text)

# ==== CRIAR ÍNDICE ====
print("🧠 Gerando embeddings com OpenAI...")
storage_context = StorageContext.from_defaults(persist_dir=OUTPUT_DIR)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# ==== SALVAR ====
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
