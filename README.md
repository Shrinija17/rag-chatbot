# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about your documents, powered by Claude (Anthropic).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red)
![Claude](https://img.shields.io/badge/LLM-Claude-orange)

## How It Works

1. **Upload** your PDF or TXT documents
2. Documents are split into chunks and converted into embeddings
3. Embeddings are stored in a ChromaDB vector database
4. When you ask a question, the most relevant chunks are retrieved
5. Claude uses those chunks to generate an accurate answer

## Tech Stack

- **LLM:** Claude (Anthropic)
- **Embeddings:** all-MiniLM-L6-v2 (HuggingFace, runs locally)
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **Frontend:** Streamlit

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
   cd rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
rag-chatbot/
├── app.py                 # Streamlit web interface
├── rag_chatbot.py         # CLI version of the chatbot
├── requirements.txt       # Python dependencies
├── .env                   # API key (not tracked by git)
├── .gitignore
├── documents/             # Place your documents here
└── README.md
```

## Features

- Upload PDF and TXT files through the web interface
- Chat-style Q&A interface
- Source attribution (shows which documents were used)
- Persistent chat history within a session
- Local embeddings (no additional API costs)
