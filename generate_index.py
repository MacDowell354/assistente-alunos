import os
from docx import Document
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_FILE = "TRANSCRIÇÃO CURSO CONSULTÓRIO HIGH TICKET.docx"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === FUNÇÃO PARA LER O .DOCX ===
def read_docx_file(file_path):
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

# === LER A TRANSCRIÇÃO ===
print("📄 Lendo transcrição...")
full_text = read_docx_file(TRANSCRIPTION_FILE)

# === DIVIDIR EM CHUNKS ===
print("✂️ Dividindo em chunks...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
chunks = parser.split_text(full_text)

# === EMBEDDINGS ===
print("🧠 Gerando embeddings...")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# === CRIAR ÍNDICE ===
documents = [chunk for chunk in chunks]
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# === SALVAR O ÍNDICE ===
print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
