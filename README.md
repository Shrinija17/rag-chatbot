<div align="center">

# 💬 DocuChat AI

**An AI-powered document Q&A chatbot using Retrieval-Augmented Generation (RAG)**

[![Python](https://img.shields.io/badge/Python-3.10+-6366f1?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-ec4899?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/LLM-Claude_AI-a855f7?style=flat-square&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Tests](https://img.shields.io/badge/Tests-6_Passing-22c55e?style=flat-square)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-6b7280?style=flat-square)](LICENSE)

Upload PDF or text documents → Ask questions → Get accurate AI-powered answers with source citations.

[Try Demo Mode](#demo-mode) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Tech Stack](#tech-stack)

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DocuChat AI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 Documents    ✂️ Chunking    🔢 Embeddings    🗄️ ChromaDB    │
│  ───────────> ───────────> ──────────────> ──────────────>      │
│  PDF / TXT     1000 char    all-MiniLM      Vector Store       │
│                chunks       -L6-v2          (In-Memory)        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                  Query Pipeline                       │      │
│  │                                                       │      │
│  │  💬 User Question                                     │      │
│  │       │                                               │      │
│  │       ▼                                               │      │
│  │  🔍 Semantic Search (Top-3 chunks from ChromaDB)      │      │
│  │       │                                               │      │
│  │       ▼                                               │      │
│  │  🧠 Claude AI (Generates answer from chunks)          │      │
│  │       │                                               │      │
│  │       ▼                                               │      │
│  │  💬 Answer + 📚 Source Citations                      │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  Frontend: Streamlit │ Backend: LangChain │ LLM: Claude        │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **LLM** | Claude (Anthropic) | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | Text → vectors (runs locally, free) |
| **Vector DB** | ChromaDB | Semantic similarity search |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Frontend** | Streamlit | Web interface |
| **Language** | Python 3.10+ | Core application |
| **Container** | Docker | Deployment |
| **Testing** | Pytest | Unit tests |

## Features

- **Document Upload** — Drag & drop PDF and TXT files
- **RAG Pipeline** — Automatic chunking, embedding, and retrieval
- **Source Citations** — See exactly which document sections were used
- **Demo Mode** — Try instantly with a pre-loaded AI Engineering guide
- **Chat Export** — Download conversation as text
- **Suggested Questions** — One-click starter questions in demo mode
- **Modular Architecture** — Clean separation of concerns (config, loader, store, chatbot)
- **Docker Support** — One command to build and run
- **Tested** — Unit tests for core modules

## Quick Start

### Option 1: Local Setup

```bash
# Clone
git clone https://github.com/Shrinija17/rag-chatbot.git
cd rag-chatbot

# Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API Key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Run
streamlit run app.py
```

### Option 2: Docker

```bash
docker build -t docuchat-ai .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key docuchat-ai
```

Then open **http://localhost:8501**

## Demo Mode

Don't have documents ready? Switch to **Demo Mode** in the sidebar to instantly try the chatbot with a pre-loaded AI Engineering guide. Ask about:

- "What is RAG and how does it work?"
- "Compare different vector databases"
- "What's the career path for an AI Engineer?"

## Project Structure

```
rag-chatbot/
├── app.py                          # Streamlit web interface
├── src/
│   ├── config.py                   # Configuration & settings
│   ├── document_loader.py          # Document loading & chunking
│   ├── vector_store.py             # Vector store creation
│   └── chatbot.py                  # RAG chain & query logic
├── tests/
│   └── test_document_loader.py     # Unit tests
├── data/
│   └── sample/                     # Demo mode documents
├── rag_chatbot.py                  # CLI version
├── Dockerfile                      # Container support
├── requirements.txt                # Dependencies
└── .env.example                    # API key template
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## How RAG Works

1. **Load** — PDF and text files are read into memory
2. **Chunk** — Documents are split into ~1000 character pieces with 200 char overlap
3. **Embed** — Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
4. **Store** — Vectors are indexed in ChromaDB for fast similarity search
5. **Retrieve** — User questions are embedded and the top 3 most similar chunks are found
6. **Generate** — Claude receives the question + relevant chunks and generates an accurate answer

---

<div align="center">

**Built by [Shrinija Kummari](https://github.com/Shrinija17)** · Powered by Claude AI

</div>
