"""Tests for the document loader module."""

import os
import tempfile
from src.document_loader import load_documents, split_documents, get_document_names


def test_load_txt_documents():
    """Test that .txt files are loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample text file
        filepath = os.path.join(tmpdir, "test.txt")
        with open(filepath, "w") as f:
            f.write("This is a test document about AI engineering.")

        docs = load_documents(tmpdir)
        assert len(docs) == 1
        assert "AI engineering" in docs[0].page_content


def test_load_empty_folder():
    """Test that empty folder returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = load_documents(tmpdir)
        assert docs == []


def test_load_nonexistent_folder():
    """Test that nonexistent folder returns empty list."""
    docs = load_documents("/nonexistent/path")
    assert docs == []


def test_split_documents():
    """Test that documents are split into chunks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")
        with open(filepath, "w") as f:
            # Write a long document that will be split
            f.write("AI engineering is important. " * 200)

        docs = load_documents(tmpdir)
        chunks = split_documents(docs)
        assert len(chunks) > 1  # Should be split into multiple chunks


def test_get_document_names():
    """Test that document names are returned correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for name in ["doc1.txt", "doc2.pdf", "image.png", "notes.txt"]:
            open(os.path.join(tmpdir, name), "w").close()

        names = get_document_names(tmpdir)
        assert "doc1.txt" in names
        assert "doc2.pdf" in names
        assert "notes.txt" in names
        assert "image.png" not in names  # Not a supported format


def test_get_document_names_empty():
    """Test empty folder returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        names = get_document_names(tmpdir)
        assert names == []
