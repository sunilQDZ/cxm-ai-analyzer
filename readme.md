# cxm-ai-analyzer

Production Voice of Customer (VOC) Analysis API with Ollama Qwen 2.5 LLM, Dynamic MySQL Database Categories, and Hybrid Rule-Based Guardrails.

## Features
- **Gibberish Detection**: Fast-path 0.36s bypass for invalid/unreadable text.
- **Ollama Qwen 2.5 Inference**: Fast, local LLM analysis for sentiment, emotion, priority, observations, and recommendations.
- **Dynamic Database Taxonomy**: Auto-syncs categories and sub-categories real-time from MySQL database.
- **Deterministic Rule Guardrails**: 100% precision alignment for complex VOC confusion pairs (App Crash, Verification Delay, Duplicate Deduction, Follow-up Issue, etc.).
- **FastAPI Endpoints**: `/health`, `/generate`, `/logs`, `/clear-cache`.

## Setup & Running
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
