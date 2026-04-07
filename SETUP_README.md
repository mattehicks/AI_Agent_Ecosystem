# VantaBlack RAG + AI_Agent_Ecosystem Setup

## Next steps (remote backend + local Web UI)

Goal: **FastAPI + Ollama run on VantaBlack** under `/mnt/llm`. **Dashboard files** run from your PC (`C:\Users\matte\Documents\GitHub\AI_Agent_Ecosystem\web`) and call the API over the LAN.

### 1. On VantaBlack — start the API

From the repo copy on the server (sync git to `/mnt/llm/AI_Agent_Ecosystem` if needed):

```bash
chmod +x /mnt/llm/AI_Agent_Ecosystem/scripts/start_vantablack_api.sh
# Ollama must already be running; FastAPI talks to http://127.0.0.1:11434 on the same machine.
/mnt/llm/AI_Agent_Ecosystem/scripts/start_vantablack_api.sh
```

Or one-liner:

```bash
cd /mnt/llm/AI_Agent_Ecosystem && source venv/bin/activate && export DISABLE_GPU_PLATFORM=1 && mkdir -p data logs && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Check from your PC:**

```powershell
curl http://192.168.0.15:8000/health
```

Use **`192.168.0.15`** when your PC is on the same Wi‑Fi/LAN as VantaBlack’s **`wlp5s0`**. Use **`192.168.2.151`** only when your PC has a **direct Ethernet** link to VantaBlack’s **`enp4s0`** (air‑gapped segment). If `curl` fails, allow **TCP 8000** on VantaBlack and confirm **`0.0.0.0:8000`** is listening (`ss -tlnp | grep 8000`).

### 2. On your PC — point the dashboard at the API

1. Edit `web\api-config.js` — default is **`http://192.168.0.15:8000`** (shared Wi‑Fi/LAN). Switch to **`http://192.168.2.151:8000`** when using the direct Ethernet link only.
2. Start a **local static server** (do not use `file://` — browsers block cross-origin requests):

```bat
START_LOCAL_WEB_UI.bat
```

3. Open **http://localhost:8080** — the UI loads from your repo; API and WebSocket use the URL in `api-config.js`.

Optional override without editing files: in the browser console run  
`localStorage.setItem('ai_ecosystem_api_base', 'http://YOUR_HOST:8000');` then reload.

### 3. When you need Ollama reachable from the PC (optional)

If you run **FastAPI on your PC** but **Ollama on VantaBlack**, Ollama must listen on the network. If **both** run on VantaBlack, the default `localhost:11434` for Ollama is enough; you only need the steps below for split “local API + remote Ollama” setups.

**On VantaBlack:**

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d/
echo '[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
ss -tlnp | grep 11434   # expect 0.0.0.0:11434
```

**Test from your PC:**

```powershell
curl http://vantablack:11434/api/tags
```

---

## Alternative: all-in-one on your PC

Run API + UI together on Windows (uses `http://localhost:8000` for both; set `web\api-config.js` to `http://localhost:8000` or temporarily point `AI_ECOSYSTEM_API_BASE` at `''` by commenting the assignment and using auto-detection — see `web\script.js`):

```powershell
cd C:\Users\matte\Documents\GitHub\AI_Agent_Ecosystem
.\venv\Scripts\activate
pip install -r requirements.txt   # if you maintain a root requirements file
python api\main.py
```

Then open `http://localhost:8000`. The batch `START_WITH_VANTABLACK.bat` also starts `python api\main.py` from the project root.

---

## System architecture (remote backend)

```
Your PC (localhost:8080)              VantaBlack (0.0.0.0:8000 + Ollama)
────────────────────────              ───────────────────────────────────
Static dashboard (web/)  ──HTTP/WS──>  FastAPI api.main:app
                                       └── Ollama @ 127.0.0.1:11434
                                       Chroma / RAG under /mnt/llm (optional)
```

---

## Files reference

**PC**

- `web\api-config.js` — API base URL for split layout  
- `START_LOCAL_WEB_UI.bat` — serve `web\` on port 8080  
- `START_WITH_VANTABLACK.bat` — start local `python api\main.py`  
- `config\vantablack_config.py` — SSH / RAG script paths for integrations  

**VantaBlack**

- `scripts/start_vantablack_api.sh` — bind FastAPI on `:8000`  
- `/mnt/llm/rag_ingest.py`, `/mnt/llm/rag_query.py` — RAG CLI  
- `/mnt/llm/chromadb/` — vector DB (if used)  
- `/mnt/llm/LLM-Models/` — GGUF files  

---

## Troubleshooting

**Dashboard says offline / fetch fails**

- Confirm `curl http://vantablack:8000/health` from the PC.  
- Confirm `web\api-config.js` matches how you reach the server (hostname or IP).  
- Use `http://` not `https://` unless you terminate TLS elsewhere.

**Port 8000 connection refused**

- On VantaBlack: `ss -tlnp | grep 8000` and firewall rules.

**Ollama only on localhost (only matters for remote-Ollama setups)**

- Use the systemd override block in section 3 above.

**RAG scripts**

```bash
python3 /mnt/llm/rag_query.py "test question" 5 YOUR_OLLAMA_MODEL_NAME
```

---

## More docs

- `INTEGRATION_GUIDE.md` — architecture notes  
- `README.md` — project overview  
