# VantaBlack Integration Config
# Run AI_Agent_Ecosystem on ALIEN1, connect to VantaBlack backend

VANTABLACK_HOST = "vantablack"  # or 192.168.0.15
VANTABLACK_OLLAMA_PORT = 11434
VANTABLACK_CHROMADB_PATH = "/mnt/llm/chromadb"

# Ollama backend endpoint (VantaBlack)
OLLAMA_BASE_URL = f"http://{VANTABLACK_HOST}:{VANTABLACK_OLLAMA_PORT}"

# SSH config for RAG operations
SSH_HOST = f"lightspeed@{VANTABLACK_HOST}"
SSH_KEY = "C:\\Users\\matte\\.ssh\\id_ed25519"

# RAG scripts on VantaBlack
RAG_INGEST_SCRIPT = "/mnt/llm/rag_ingest.py"
RAG_QUERY_SCRIPT = "/mnt/llm/rag_query.py"

# Local model discovery
LOCAL_MODEL_PATH = None  # All models on VantaBlack via Ollama
