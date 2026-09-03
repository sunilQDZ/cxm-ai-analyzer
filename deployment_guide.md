# 🚀 Quick Live Deployment Guide

A simple, step-by-step guide to deploy the CX Qwen VOC Analysis API on your live server .

---

## ⚡ Step 1: Install Ollama & Download Model

Run in your terminal:
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama and download Qwen model
ollama serve &
ollama pull qwen2.5:1.5b
```

---

## 📦 Step 2: Install Project Dependencies

Navigate to project directory and install requirements:
```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Step 3: Configure Environment Variables

Edit `config.py` or create a `.env` file in the project root:

```ini
API_TOKEN=my_secret_123
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT=90
OLLAMA_NUM_THREADS=4
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=300
OLLAMA_KEEP_ALIVE=10m
BATCH_MAX_WORKERS=1

MYSQL_HOST=188.241.187.49
MYSQL_USER=surveycx_admin
MYSQL_PASSWORD=SurveyCX@2026
MYSQL_DB=surveycx_demo
```

---

## 🚀 Step 4: Run the Server in Background (Systemd)

### 1. Ollama Service Configuration (with CPU limits)
Create or edit `/etc/systemd/system/ollama.service`:
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3s
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=30m"

[Install]
WantedBy=default.target
```

### 2. CX API Service Configuration
Create `/etc/systemd/system/cx-qwen.service`:
```bash
sudo nano /etc/systemd/system/cx-qwen.service
```

Paste:
```ini
[Unit]
Description=CX Qwen VOC Analysis API
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/cx_qwen_api
ExecStart=/var/www/cx_qwen_api/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3s

[Install]
WantedBy=multi-user.target
```

Start & enable on boot:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cx-qwen
```

*(Alternatively, to run manually: `python app.py`)*

---

## ✅ Step 5: Verify Deployment

### 1. Check Health & Dependencies
```bash
curl http://localhost:8000/health
```
**Response**: `{"status": "ok", "ollama_available": true, "database_connected": true}`

### 2. Test Live VOC Inference
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -H "x-api-key: my_secret_123" \
     -d '{"data": [{"id": "1", "comments": "Agent was very polite and helpful"}]}'
```

### 3. Check Live Logs
```bash
curl "http://localhost:8000/logs?limit=10" -H "x-api-key: my_secret_123"
```
