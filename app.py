import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA

load_dotenv()

# ============ Page Config ============

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============ Custom Styling ============

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.85rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============ Helper Functions ============

@st.cache_resource
def load_and_process_documents(folder_path):
    """Load documents, split into chunks, and create vector store."""
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        return None

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store with free local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store


def get_chatbot(vector_store):
    """Create the RAG chatbot."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=1024
    )

    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return chatbot

# ============ Sidebar: File Upload ============

with st.sidebar:
    st.header("📄 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # Save uploaded files
    if uploaded_files:
        os.makedirs("./documents", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("./documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s)!")

    # Show current documents
    if os.path.exists("./documents"):
        files = [f for f in os.listdir("./documents") if f.endswith(('.pdf', '.txt'))]
        if files:
            st.markdown("**Loaded documents:**")
            for f in files:
                st.markdown(f"- {f}")

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("- 🧠 Claude (Anthropic)")
    st.markdown("- 🔗 LangChain")
    st.markdown("- 📊 ChromaDB")
    st.markdown("- 🎈 Streamlit")

# ============ Main App ============

st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents — powered by Claude</div>', unsafe_allow_html=True)

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Please set your ANTHROPIC_API_KEY in the .env file")
    st.stop()

# Check for documents
documents_folder = "./documents"
if not os.path.exists(documents_folder):
    os.makedirs(documents_folder)

files = [f for f in os.listdir(documents_folder) if f.endswith(('.pdf', '.txt'))]

if not files:
    st.info("👈 Upload some documents using the sidebar to get started!")
    st.stop()

# Load documents and create vector store
with st.spinner("Loading documents and creating embeddings..."):
    vector_store = load_and_process_documents(documents_folder)

if vector_store is None:
    st.error("No documents could be loaded.")
    st.stop()

chatbot = get_chatbot(vector_store)

# ============ Chat Interface ============

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke({"query": prompt})
            answer = result["result"]

            st.markdown(answer)

            # Show sources
            if result.get("source_documents"):
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        source = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Source {i+1}:** {source}")
                        st.markdown(f"```\n{doc.page_content[:300]}...\n```")

    st.session_state.messages.append({"role": "assistant", "content": answer})
