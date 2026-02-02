"""Document loading and processing module."""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}


def load_documents(folder_path: str) -> list:
    """Load all supported documents from a folder."""
    documents = []

    if not os.path.exists(folder_path):
        return documents

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            file_path = os.path.join(folder_path, filename)
            loader = SUPPORTED_EXTENSIONS[ext](file_path)
            documents.extend(loader.load())

    return documents


def split_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_document_names(folder_path: str) -> list:
    """Get list of supported document filenames in folder."""
    if not os.path.exists(folder_path):
        return []
    return [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
