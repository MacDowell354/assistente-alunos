import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# === CONFIGURAÇÃO ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TXT_FILE = "transcricoes.txt"
OUTPUT_DIR = "storage"

# === VERIFICA A CHAVE ===
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# === PREPARAR DOCUMENTO ===
if not os.path.exists(TXT_FILE):
    raise FileNotFoundError(f"O arquivo {TXT_FILE} não foi encontrado.")

# Copia o txt para uma pasta temporária
os.makedirs("tmp", exist_ok=True)
os.system(f"copy {TXT_FILE} tmp\\conteudo.txt" if os.name == "nt" else f"cp {TXT_FILE} tmp/conteudo.txt")

# === CRIAR ÍNDICE ===
print("📄 Lendo conteúdo...")
documents = SimpleDirectoryReader("tmp").load_data()

print("🧠 Gerando embeddings com OpenAI...")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

print("🔍 Criando índice vetorial...")
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

print("💾 Salvando índice em:", OUTPUT_DIR)
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print("✅ Índice gerado com sucesso!")
