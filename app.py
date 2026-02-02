import streamlit as st
import os
import hashlib
import shutil
from dotenv import load_dotenv
from src.document_loader import load_documents, split_documents, get_document_names
from src.vector_store import create_vector_store
from src.chatbot import create_chatbot, ask
from src.config import DOCUMENTS_DIR, SAMPLE_DIR

load_dotenv()

if "ANTHROPIC_API_KEY" not in os.environ:
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

# ============ Page Config ============

st.set_page_config(
    page_title="DocuChat AI — RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Styling ============

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0a0f; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111118;
        border-right: 1px solid #1e1e2e;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li { color: #b0b0c0; }

    /* Hero */
    .hero { text-align: center; padding: 1.2rem 0 0.3rem 0; }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa; font-size: 0.72rem; font-weight: 600;
        padding: 4px 14px; border-radius: 50px;
        letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.4rem; font-weight: 800; color: #fff;
        margin-bottom: 0.3rem; letter-spacing: -0.5px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-desc { color: #6b7280; font-size: 1rem; max-width: 520px; margin: 0 auto 1rem auto; line-height: 1.6; }

    /* Architecture */
    .arch-container {
        background: #111118; border: 1px solid #1e1e2e; border-radius: 16px;
        padding: 1.5rem; margin: 1rem auto; max-width: 700px; text-align: center;
    }
    .arch-title { color: #a78bfa; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 1rem; }
    .arch-flow {
        display: flex; align-items: center; justify-content: center;
        gap: 0.4rem; flex-wrap: wrap; font-size: 0.8rem;
    }
    .arch-step {
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
        border-radius: 10px; padding: 8px 14px; color: #c4b5fd; font-weight: 500;
    }
    .arch-arrow { color: #6366f1; font-size: 1.2rem; }

    /* Stats */
    .stats-row { display: flex; justify-content: center; gap: 0.8rem; margin: 0.8rem 0 1.2rem 0; }
    .stat-card {
        background: #111118; border: 1px solid #1e1e2e; border-radius: 12px;
        padding: 10px 22px; text-align: center; min-width: 100px;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #6366f1; }
    .stat-name { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

    /* Steps cards */
    .steps-container { display: flex; justify-content: center; gap: 1rem; margin: 1.5rem auto; max-width: 700px; flex-wrap: wrap; }
    .step-card {
        background: #111118; border: 1px solid #1e1e2e; border-radius: 14px;
        padding: 1.3rem; width: 190px; text-align: center; transition: border-color 0.3s;
    }
    .step-card:hover { border-color: #6366f1; }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 50%; font-size: 0.75rem; font-weight: 700; color: #fff; margin-bottom: 0.5rem;
    }
    .step-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .step-title { font-size: 0.9rem; font-weight: 600; color: #e5e7eb; margin-bottom: 0.2rem; }
    .step-desc { font-size: 0.75rem; color: #6b7280; line-height: 1.4; }

    /* Chat */
    .stChatMessage { border-radius: 12px !important; border: 1px solid #1e1e2e !important; background: #111118 !important; margin-bottom: 0.6rem !important; }
    .stChatInput > div { border-color: #1e1e2e !important; background: #111118 !important; border-radius: 12px !important; }

    /* Sources */
    .src-card {
        background: #0d0d14; border: 1px solid #1e1e2e; border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0; padding: 8px 12px; margin: 4px 0; font-size: 0.8rem; color: #9ca3af;
    }
    .src-card strong { color: #c4b5fd; }

    /* Sidebar elements */
    .sidebar-section { font-size: 0.68rem; font-weight: 700; color: #6366f1; text-transform: uppercase; letter-spacing: 1.5px; margin: 1rem 0 0.5rem 0; }
    .sidebar-divider { border: none; border-top: 1px solid #1e1e2e; margin: 0.8rem 0; }
    .file-item {
        display: flex; align-items: center; gap: 8px;
        background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.12);
        border-radius: 8px; padding: 6px 10px; margin-bottom: 4px; color: #d1d5db; font-size: 0.8rem;
    }
    .tech-pills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 0.4rem; }
    .pill {
        background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
        color: #818cf8; font-size: 0.7rem; font-weight: 500; padding: 3px 9px; border-radius: 50px;
    }
    .demo-badge {
        display: inline-block; background: rgba(234,179,8,0.1); border: 1px solid rgba(234,179,8,0.3);
        color: #fbbf24; font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
        border-radius: 50px; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============ Helpers ============

def get_doc_hash(folder):
    files = sorted(get_document_names(folder))
    return hashlib.md5("".join(files).encode()).hexdigest()


def process_documents(folder):
    docs = load_documents(folder)
    if not docs:
        return None, 0, []
    chunks = split_documents(docs)
    vs = create_vector_store(chunks)
    names = get_document_names(folder)
    return vs, len(chunks), names


# ============ Sidebar ============

with st.sidebar:
    st.markdown("## 💬 DocuChat AI")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Mode selection
    st.markdown('<div class="sidebar-section">🎯 Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        ["📄 My Documents", "🎮 Demo Mode"],
        label_visibility="collapsed",
        help="Demo mode lets you try the chatbot with a pre-loaded AI Engineering guide"
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if mode == "📄 My Documents":
        st.markdown('<div class="sidebar-section">📁 Upload Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop files here", type=["pdf", "txt"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploaded_files:
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(DOCUMENTS_DIR, f.name), "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
            st.session_state.doc_hash = None
            st.session_state.vector_store = None
            st.rerun()
    else:
        st.markdown('<span class="demo-badge">🎮 DEMO MODE</span>', unsafe_allow_html=True)
        st.markdown("Using pre-loaded **AI Engineering Guide** — ask anything about AI engineering, RAG, vector databases, or career paths!")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Show loaded docs
    active_folder = DOCUMENTS_DIR if mode == "📄 My Documents" else SAMPLE_DIR
    files = get_document_names(active_folder)
    if files:
        st.markdown(f'<div class="sidebar-section">📂 {len(files)} Document(s)</div>', unsafe_allow_html=True)
        for f in files:
            icon = "📕" if f.endswith('.pdf') else "📄"
            st.markdown(f'<div class="file-item">{icon} {f}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📋 Export", use_container_width=True):
            if st.session_state.get("messages"):
                chat_text = ""
                for m in st.session_state.messages:
                    role = "You" if m["role"] == "user" else "DocuChat AI"
                    chat_text += f"{role}: {m['content']}\n\n"
                st.download_button("⬇️ Download", chat_text, "docuchat_export.txt", use_container_width=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">⚡ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-pills">
        <span class="pill">Claude AI</span><span class="pill">LangChain</span>
        <span class="pill">ChromaDB</span><span class="pill">Streamlit</span>
        <span class="pill">HuggingFace</span><span class="pill">Python</span>
    </div>
    """, unsafe_allow_html=True)


# ============ Main ============

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Document Assistant</div>
    <div class="hero-title">Chat with your <span>Documents</span></div>
    <div class="hero-desc">Upload any PDF or text file and get instant, accurate answers powered by Claude AI and Retrieval-Augmented Generation.</div>
</div>
""", unsafe_allow_html=True)

# Architecture diagram
st.markdown("""
<div class="arch-container">
    <div class="arch-title">⚙️ How It Works — RAG Architecture</div>
    <div class="arch-flow">
        <div class="arch-step">📄 Documents</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">✂️ Chunking</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">🔢 Embeddings</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">🗄️ ChromaDB</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">🔍 Retrieval</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">🧠 Claude AI</div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">💬 Answer</div>
    </div>
</div>
""", unsafe_allow_html=True)

# API key check
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("⚠️ ANTHROPIC_API_KEY not found. Add it to your `.env` file or Streamlit secrets.")
    st.stop()

# Document handling
active_folder = DOCUMENTS_DIR if mode == "📄 My Documents" else SAMPLE_DIR
os.makedirs(active_folder, exist_ok=True)
files = get_document_names(active_folder)

if not files:
    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-num">1</div><div class="step-icon">📄</div>
            <div class="step-title">Upload</div>
            <div class="step-desc">Add PDF or TXT files via the sidebar</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div><div class="step-icon">💬</div>
            <div class="step-title">Ask</div>
            <div class="step-desc">Type any question about your docs</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div><div class="step-icon">✨</div>
            <div class="step-title">Get Answers</div>
            <div class="step-desc">AI answers with source citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Upload documents or switch to **Demo Mode** in the sidebar!")
    st.stop()

# Process documents
current_hash = get_doc_hash(active_folder) + mode
need_reload = st.session_state.get("doc_hash") != current_hash

if need_reload:
    with st.spinner("🔄 Processing documents..."):
        vs, n_chunks, names = process_documents(active_folder)
        if vs:
            st.session_state.vector_store = vs
            st.session_state.num_chunks = n_chunks
            st.session_state.doc_names = names
            st.session_state.doc_hash = current_hash

vs = st.session_state.get("vector_store")
if vs is None:
    st.error("Could not process documents.")
    st.stop()

# Stats
n_docs = len(st.session_state.get("doc_names", []))
n_chunks = st.session_state.get("num_chunks", 0)
n_qs = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card"><div class="stat-value">{n_docs}</div><div class="stat-name">Documents</div></div>
    <div class="stat-card"><div class="stat-value">{n_chunks}</div><div class="stat-name">Chunks</div></div>
    <div class="stat-card"><div class="stat-value">{n_qs}</div><div class="stat-name">Questions</div></div>
</div>
""", unsafe_allow_html=True)

chatbot = create_chatbot(vs)

# ============ Chat ============

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, s in enumerate(msg["sources"]):
                    st.markdown(f'<div class="src-card"><strong>Source {i+1}</strong> — {s["name"]}<br>{s["preview"]}</div>', unsafe_allow_html=True)

# Suggested questions in demo mode
if mode == "🎮 Demo Mode" and not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(3)
    suggestions = ["What is RAG?", "Compare vector databases", "AI Engineer career path"]
    for i, q in enumerate(suggestions):
        if cols[i].button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# Handle suggested question
prompt = st.session_state.pop("pending_question", None)

if prompt is None:
    prompt = st.chat_input("Ask anything about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            result = ask(chatbot, prompt)
            st.markdown(result["answer"])

            sources = []
            if result["sources"]:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(result["sources"]):
                        name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        preview = doc.page_content[:200] + "..."
                        st.markdown(f'<div class="src-card"><strong>Source {i+1}</strong> — {name}<br>{preview}</div>', unsafe_allow_html=True)
                        sources.append({"name": name, "preview": preview})

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": sources})
    st.rerun()
