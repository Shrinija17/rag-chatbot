import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA

load_dotenv()

if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

# ============ Page Config ============

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Styling ============

st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Reset & Global */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0a0a0f;
    }

    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111118;
        border-right: 1px solid #1e1e2e;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #b0b0c0;
    }

    /* Hero */
    .hero-container {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 50px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        color: #6b7280;
        font-size: 1.05rem;
        max-width: 500px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* Stats */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .stat-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 14px;
        padding: 14px 28px;
        text-align: center;
        min-width: 120px;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #6366f1;
    }
    .stat-name {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 2px;
    }

    /* Getting started cards */
    .steps-container {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin: 2rem auto;
        max-width: 700px;
        flex-wrap: wrap;
    }
    .step-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 16px;
        padding: 1.5rem;
        width: 200px;
        text-align: center;
        transition: border-color 0.3s;
    }
    .step-card:hover {
        border-color: #6366f1;
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 50%;
        font-size: 0.85rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.7rem;
    }
    .step-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-size: 0.78rem;
        color: #6b7280;
        line-height: 1.4;
    }

    /* Chat area */
    .stChatMessage {
        border-radius: 14px !important;
        border: 1px solid #1e1e2e !important;
        background: #111118 !important;
        margin-bottom: 0.8rem !important;
    }

    /* Chat input */
    .stChatInput > div {
        border-color: #1e1e2e !important;
        background: #111118 !important;
        border-radius: 14px !important;
    }

    /* Source card */
    .src-card {
        background: #0d0d14;
        border: 1px solid #1e1e2e;
        border-left: 3px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #9ca3af;
    }
    .src-card strong {
        color: #c4b5fd;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* Doc file item */
    .file-item {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
        color: #d1d5db;
        font-size: 0.82rem;
    }
    .file-icon {
        font-size: 1.1rem;
    }

    /* Tech stack pills */
    .tech-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 0.5rem;
    }
    .pill {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        color: #818cf8;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 50px;
    }

    /* Divider */
    .sidebar-divider {
        border: none;
        border-top: 1px solid #1e1e2e;
        margin: 1rem 0;
    }

    /* Info box override */
    .stAlert {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        color: #9ca3af;
    }

    /* Spinner */
    .stSpinner > div {
        color: #a78bfa !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111118 !important;
        border-radius: 10px !important;
    }

    /* Clear chat button */
    .clear-btn {
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

def get_documents_hash(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt'))])
    return hashlib.md5("".join(files).encode()).hexdigest()


def load_and_process_documents(folder_path):
    documents = []
    doc_names = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            doc_names.append(filename)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            doc_names.append(filename)

    if not documents:
        return None, 0, []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vector_store, len(chunks), doc_names


def get_chatbot(vector_store):
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


# ============ Sidebar ============

with st.sidebar:
    st.markdown("## 💬 DocuChat AI")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="sidebar-section">📁 Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        os.makedirs("./documents", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("./documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
        # Force reprocessing of all documents
        st.session_state.doc_hash = None
        st.session_state.vector_store = None
        st.rerun()

    # Show loaded documents
    if os.path.exists("./documents"):
        files = [f for f in os.listdir("./documents") if f.endswith(('.pdf', '.txt'))]
        if files:
            for f in files:
                icon = "📕" if f.endswith('.pdf') else "📄"
                st.markdown(f'<div class="file-item"><span class="file-icon">{icon}</span>{f}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="sidebar-section">⚡ Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-pills">
        <span class="pill">Claude AI</span>
        <span class="pill">LangChain</span>
        <span class="pill">ChromaDB</span>
        <span class="pill">Streamlit</span>
        <span class="pill">HuggingFace</span>
    </div>
    """, unsafe_allow_html=True)


# ============ Main Content ============

# Hero
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-Powered Document Assistant</div>
    <div class="hero-title">Chat with your <span>Documents</span></div>
    <div class="hero-desc">Upload PDFs or text files and get instant, accurate answers powered by Claude AI and RAG technology.</div>
</div>
""", unsafe_allow_html=True)

# Check API key
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("⚠️ ANTHROPIC_API_KEY not found. Add it to your .env file.")
    st.stop()

# Check documents
documents_folder = "./documents"
os.makedirs(documents_folder, exist_ok=True)
files = [f for f in os.listdir(documents_folder) if f.endswith(('.pdf', '.txt'))]

if not files:
    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-icon">📄</div>
            <div class="step-title">Upload</div>
            <div class="step-desc">Add your PDF or TXT files using the sidebar</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-icon">💬</div>
            <div class="step-title">Ask</div>
            <div class="step-desc">Type any question about your documents</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-icon">✨</div>
            <div class="step-title">Get Answers</div>
            <div class="step-desc">Receive AI-powered answers with sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Process documents
current_hash = get_documents_hash(documents_folder)
need_reload = st.session_state.get("doc_hash") != current_hash

if need_reload:
    with st.spinner("🔄 Processing your documents..."):
        vector_store, num_chunks, doc_names = load_and_process_documents(documents_folder)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.num_chunks = num_chunks
            st.session_state.doc_names = doc_names
            st.session_state.doc_hash = current_hash

vector_store = st.session_state.get("vector_store")

if vector_store is None:
    st.error("Could not process documents. Please try uploading again.")
    st.stop()

# Stats
num_docs = len(st.session_state.get("doc_names", []))
num_chunks = st.session_state.get("num_chunks", 0)
num_questions = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">{num_docs}</div>
        <div class="stat-name">Documents</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{num_chunks}</div>
        <div class="stat-name">Chunks</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{num_questions}</div>
        <div class="stat-name">Questions</div>
    </div>
</div>
""", unsafe_allow_html=True)

chatbot = get_chatbot(vector_store)

# ============ Chat ============

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(
                        f'<div class="src-card"><strong>Source {i+1}</strong> — {src["name"]}<br>{src["preview"]}</div>',
                        unsafe_allow_html=True
                    )

# Input
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke({"query": prompt})
            answer = result["result"]
            st.markdown(answer)

            sources = []
            if result.get("source_documents"):
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        preview = doc.page_content[:200] + "..."
                        st.markdown(
                            f'<div class="src-card"><strong>Source {i+1}</strong> — {name}<br>{preview}</div>',
                            unsafe_allow_html=True
                        )
                        sources.append({"name": name, "preview": preview})

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()
