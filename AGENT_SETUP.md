# Agent Setup Guide

## Overview

The project now supports two analysis modes:

1. Simple mode, which is deterministic and free.
2. Tool-calling mode, which uses a local Ollama model through LangChain.

## Ollama Setup

Ollama runs locally, so there is no API bill.

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama service.
3. Pull a model, for example:

```bash
ollama pull llama3.1
```

Optional environment variables:

```env
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

On Windows PowerShell:

```powershell
$env:OLLAMA_MODEL = "llama3.1"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

## Dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

This includes `langchain-core`, `langgraph`, and `langchain-ollama`.

## Backend

Start the API server:

```bash
uvicorn backend.main:app --reload
```

The analysis endpoint is `POST /api/analyze`.

## Frontend

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The upload UI now behaves like a chat interface and can switch between simple mode and Ollama-backed tool calling.

## Troubleshooting

If tool-calling mode says Ollama is not configured, verify that:

1. Ollama is running locally.
2. A model has been pulled.
3. `OLLAMA_BASE_URL` matches your local service.

If you want zero-cost operation, use simple mode only.
