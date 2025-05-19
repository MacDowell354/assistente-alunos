import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.storage.storage_context import StorageContext

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TXT_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === PREPARA DOCUMENTO ===
if not os.path.exists(TXT_FILE):
    raise FileNotFoundError(f"O arquivo {TXT_FILE} não foi encontrado.")

# Cria pasta temporária
os.makedirs("tmp", exist_ok=True)
os.system(f'cp "{TXT_FILE}" tmp/conteudo.txt' if os.name == "nt" else f'cp "{TXT_FILE}" tmp/conteudo.txt')

# === LÊ DOCUMENTOS ===
print("📄 Lendo conteúdo do arquivo...")
documents = SimpleDirectoryReader("tmp").load_data()

# === EMBEDDINGS ===
print("🧠 Gerando embeddings com OpenAI...")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# === CRIA ÍNDICE ===
print("📦 Criando índice vetorial...")
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# === SALVA ===
print(f"💾 Salvando índice em: {OUTPUT_DIR}")
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
