import streamlit as st
import os
import sqlite3
import uuid
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

# New Imports for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to .env or Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DATABASE & VECTOR MANAGER ---
class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_message(self, session_id, role, content):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
            (session_id, role, content)
        )
        self.conn.commit()

    def get_history(self, session_id):
        cursor = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", 
            (session_id,)
        )
        return cursor.fetchall()

    def get_all_sessions(self):
        cursor = self.conn.execute("SELECT DISTINCT session_id FROM messages ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]

db = DatabaseManager()

# --- 3. RAG LOGIC (CHUNKING & SIMILARITY) ---
class VectorManager:
    def __init__(self):
        # Local embeddings (Free, runs on your CPU)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def get_vector_store(self, uploaded_files, local_folder="documents"):
        all_text = ""
        
        # 1. Extract from local folder
        if os.path.exists(local_folder):
            for f in os.listdir(local_folder):
                all_text += extract_text(os.path.join(local_folder, f), is_path=True) + "\n"
        
        # 2. Extract from uploads
        if uploaded_files:
            for f in uploaded_files:
                all_text += extract_text(f) + "\n"

        if not all_text.strip():
            return None

        # 3. Create Chunks
        chunks = self.text_splitter.split_text(all_text)
        
        # 4. Build FAISS index (Uses Cosine Similarity via normalized embeddings)
        vector_db = FAISS.from_texts(chunks, self.embeddings)
        return vector_db

vm = VectorManager()

# --- 4. DATA EXTRACTION ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        name = file_source if is_path else file_source.name
        ext = os.path.splitext(name)[-1].lower()
        if ext == ".pdf":
            return "\n".join([p.extract_text() for p in PdfReader(file_source).pages if p.extract_text()])
        elif ext == ".docx":
            return "\n".join([p.text for p in docx.Document(file_source).paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f: return f.read()
            return file_source.getvalue().decode("utf-8", errors="ignore")
    except: return ""
    return ""

# --- 5. SESSION MANAGEMENT ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("Research Control")
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.subheader("Previous Chats")
    sessions = db.get_all_sessions()
    if sessions:
        selected_session = st.selectbox(
            "Select a chat:", sessions,
            index=sessions.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in sessions else 0
        )
        if selected_session != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_session
            st.rerun()

    st.divider()
    uploads = st.file_uploader("Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 7. MAIN INTERFACE ---
st.title("🤖 AI Research Assistant")

# Display History
for role, content in db.get_history(st.session_state.current_chat_id):
    with st.chat_message(role):
        st.markdown(content)

# CHAT INPUT
user_input = st.chat_input("Ask about your documents...")

if user_input:
    db.save_message(st.session_state.current_chat_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # 1. Perform Similarity Search
        vector_db = vm.get_vector_store(uploads)
        context = "No relevant document context found."
        
        if vector_db:
            # Search for top 4 most similar chunks
            docs = vector_db.similarity_search(user_input, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build Prompt
        prompt = (
            "You are a helpful research assistant. Use the provided CONTEXT to answer the user.\n"
            "If the context doesn't contain the answer, use your general knowledge but mention it.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {user_input}"
        )

        try:
            full_res = ""
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            db.save_message(st.session_state.current_chat_id, "assistant", full_res)
            
        except Exception as e:
            st.error(f"Error: {e}")
