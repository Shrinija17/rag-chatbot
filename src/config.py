"""Configuration for the RAG Chatbot."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model settings
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 3

# Paths
DOCUMENTS_DIR = "./documents"
SAMPLE_DIR = "./data/sample"
