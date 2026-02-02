"""Chatbot module - creates the RAG chain."""

from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K


def create_chatbot(vector_store: Chroma) -> RetrievalQA:
    """Create a RAG chatbot from a vector store."""
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
    )


def ask(chatbot: RetrievalQA, question: str) -> dict:
    """Ask a question and return the result with sources."""
    result = chatbot.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result.get("source_documents", []),
    }
