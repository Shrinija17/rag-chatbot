"""Vector store management module."""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def create_vector_store(chunks: list) -> Chroma:
    """Create an in-memory vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)
