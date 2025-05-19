import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.llms import OpenAI

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === FUNÇÃO PARA LER .TXT ===
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# === LER A TRANSCRIÇÃO ===
print("📄 Lendo conteúdo do arquivo...")
full_text = read_txt_file(TRANSCRIPTION_FILE)

# === DIVIDIR EM CHUNKS ===
print("✂️ Dividindo em trechos...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([SimpleDirectoryReader.from_texts([full_text]).load_data()[0]])

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# === CRIAR DOCUMENTOS E ÍNDICE ===
index = VectorStoreIndex(nodes)

# === SALVAR O ÍNDICE ===
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
