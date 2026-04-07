# AI_Agent_Ecosystem + VantaBlack RAG Integration

## Architecture

```
ALIEN1 (Windows 10)                    VantaBlack (Ubuntu)
─────────────────────                  ────────────────────
┌─────────────────┐                    ┌─────────────────┐
│ Web UI          │                    │ Ollama LLM      │
│ localhost:8000  │────SSH────────────>│ (dual 3090s)    │
│                 │                    │                 │
│ AI Agent        │                    │ ChromaDB        │
│ Ecosystem       │                    │ Vector Store    │
│                 │                    │                 │
│ FastAPI Backend │<───────────────────│ Model Inference │
└─────────────────┘                    └─────────────────┘
```

## Features Combined

### From AI_Agent_Ecosystem:
- Beautiful web UI with research categories
- Document batch processing
- Agent orchestration
- Task management
- WebSocket monitoring

### From VantaBlack RAG:
- 48GB VRAM (dual RTX 3090)
- ChromaDB semantic search
- Legal document processing
- 30B+ parameter model support
- GPU-accelerated embeddings

### New "Document RAG" Category:
- Query legal/technical documents
- Semantic search with citations
- Model selection (qwen2.5:32b, wizard-vicuna:30b, etc.)
- Real-time ingestion
- Database statistics

## Quick Start

### 1. Start VantaBlack Services

SSH into VantaBlack:
```powershell
ssh -i C:\Users\matte\.ssh\id_ed25519 lightspeed@vantablack
```

Ensure Ollama is running:
```bash
ollama list  # Check loaded models
# If empty, load a model (see VANTABLACK_QUICKSTART.md)
```

### 2. Start AI_Agent_Ecosystem on ALIEN1

```powershell
cd C:\Users\matte\Documents\GitHub\AI_Agent_Ecosystem
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python api/main.py
```

### 3. Access Web UI

Open browser: `http://localhost:8000`

Navigate to **Research** → **Document RAG**

## Configuration

### VantaBlack Connection

Edit `config/vantablack_config.py`:
```python
VANTABLACK_HOST = "vantablack"  # or 192.168.0.15
OLLAMA_BASE_URL = "http://vantablack:11434"
SSH_HOST = "lightspeed@vantablack"
```

### Ollama Backend

Already configured to use VantaBlack by default:
- `llm_backends/ollama_backend.py` points to `http://vantablack:11434`
- All model inference happens on VantaBlack GPUs

## Usage Examples

### Web UI - Document RAG

1. **Query Documents**:
   - Category: Document RAG
   - Input: "What are the liability clauses in section 3?"
   - Model: qwen2.5:32b (or wizard-vicuna:30b)
   - Click "Generate"

2. **Ingest PDFs**:
   - Upload PDFs to VantaBlack: `/mnt/llm/contracts/`
   - Use ingestion endpoint
   - Documents auto-chunked and indexed

3. **Check Status**:
   - View document count
   - See available models
   - Monitor ChromaDB health

### API Endpoints

**Query RAG**:
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the indemnification clause?", "model": "qwen2.5:32b"}'
```

**Ingest Documents**:
```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "/mnt/llm/contracts"}'
```

**Status**:
```bash
curl http://localhost:8000/rag/status
```

## Available Models

Your VantaBlack `/mnt/llm/LLM-Models`:
- **Wizard-Vicuna 30B** (18GB) - Best for complex legal reasoning
- **Llama 3.1 8B Abliterated** - Fast, uncensored
- **Dolphin 2.9.4** - Instruction-tuned
- **qwen2.5:32b** - Currently downloading

Load models with:
```bash
/mnt/llm/load_models.sh
```

## Technical Stack

**Frontend (ALIEN1)**:
- FastAPI backend
- HTML/JS/CSS web UI
- Agent orchestration

**Backend (VantaBlack)**:
- Ollama (LLM inference)
- ChromaDB (vector database)
- sentence-transformers (embeddings)
- Python RAG scripts

## File Locations

### ALIEN1
```
C:\Users\matte\Documents\GitHub\AI_Agent_Ecosystem\
├── agents\rag_agent.py          # New RAG agent
├── config\vantablack_config.py  # VantaBlack settings
├── llm_backends\ollama_backend.py  # Updated for remote
└── web\                         # UI files
```

### VantaBlack
```
/mnt/llm/
├── rag_ingest.py
├── rag_query.py
├── chromadb/
└── LLM-Models/
```

## Troubleshooting

**"Connection refused" to VantaBlack**:
```powershell
# Test connectivity
ssh lightspeed@vantablack "ollama list"
```

**Web UI won't start**:
```powershell
cd C:\Users\matte\Documents\GitHub\AI_Agent_Ecosystem
python api/main.py  # Check error output
```

**RAG queries fail**:
```bash
# On VantaBlack, test directly:
python3 /mnt/llm/rag_query.py "test question"
```

**Ollama models not loading**:
```bash
# Check Ollama service
systemctl status ollama
# Restart if needed
sudo systemctl restart ollama
```

## Next Steps

1. ✅ VantaBlack RAG pipeline operational
2. ✅ MCP server for Cursor ready
3. 🔄 Add RAG to web UI categories
4. 🔄 Test end-to-end workflow
5. 🔜 Deploy production config
6. 🔜 Add conversation memory
7. 🔜 Implement reranking

---

**System Status**: ✅ Ready for testing

Run `python api/main.py` on ALIEN1, open `http://localhost:8000`, and start querying your documents!
