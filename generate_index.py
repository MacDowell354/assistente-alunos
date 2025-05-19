import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext

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
nodes = parser.get_nodes_from_documents([Document(text=full_text)])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# === CRIAR DOCUMENTOS E ÍNDICE ===
index = VectorStoreIndex.from_documents(nodes, service_context=service_context)

# === SALVAR O ÍNDICE ===
print(f"💾 Salvando índice em: {OUTPUT_DIR}")
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
