# RAG Chatbot - Using Claude (Anthropic)
# This chatbot answers questions based on your documents

# ============ STEP 1: Import Libraries ============

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Free embeddings!
from langchain_anthropic import ChatAnthropic  # Claude!
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables (like your API key)
load_dotenv()

# ============ STEP 2: Load Documents ============

def load_documents(folder_path):
    """
    Load all PDF and text files from a folder.
    """
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"  Loaded: {filename}")

        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"  Loaded: {filename}")

    return documents

# ============ STEP 3: Split Documents into Chunks ============

def split_documents(documents):
    """
    Split documents into smaller chunks.
    - chunk_size: how many characters per chunk
    - chunk_overlap: overlap between chunks (helps with context)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")
    return chunks

# ============ STEP 4: Create Vector Store ============

def create_vector_store(chunks):
    """
    Create a vector database from document chunks.
    Using free HuggingFace embeddings (runs locally, no API needed!)
    """
    print("  Loading embedding model (first time may take a moment)...")

    # This embedding model is free and runs locally on your computer
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast, and good quality
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("  Vector store created!")
    return vector_store

# ============ STEP 5: Create the Chatbot ============

def create_chatbot(vector_store):
    """
    Create a chatbot using Claude to answer questions.
    """
    # Claude as our LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # You can also use claude-3-haiku for cheaper
        temperature=0,
        max_tokens=1024
    )

    # Create a retrieval chain
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}  # Return top 3 relevant chunks
        ),
        return_source_documents=True
    )

    return chatbot

# ============ STEP 6: Main Function ============

def main():
    print("=" * 50)
    print("  RAG Chatbot (Powered by Claude)")
    print("=" * 50)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: Please set your ANTHROPIC_API_KEY in the .env file")
        print("Get your API key from: https://console.anthropic.com/")
        return

    documents_folder = "./documents"

    # Check for documents
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)

    files = [f for f in os.listdir(documents_folder) if f.endswith(('.pdf', '.txt'))]
    if not files:
        print(f"\nNo documents found in {documents_folder}")
        print("Please add some PDF or TXT files to that folder.")
        return

    print("\n[1/4] Loading documents...")
    documents = load_documents(documents_folder)

    print("\n[2/4] Splitting into chunks...")
    chunks = split_documents(documents)

    print("\n[3/4] Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("\n[4/4] Creating chatbot...")
    chatbot = create_chatbot(vector_store)

    # Chat loop
    print("\n" + "=" * 50)
    print("  Ready! Ask me anything about your documents.")
    print("  Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        # Get answer
        result = chatbot.invoke({"query": question})

        print(f"\nClaude: {result['result']}\n")

if __name__ == "__main__":
    main()
