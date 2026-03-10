import streamlit as st
import os
import sqlite3
import uuid
import docx
import time
from google import genai
from google.api_core import exceptions
from pypdf import PdfReader
from dotenv import load_dotenv

# Modern LangChain imports for 2026
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DATABASE (Chat History) ---
class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, role TEXT, content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_message(self, session_id, role, content):
        self.conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        self.conn.commit()

    def get_history(self, session_id):
        cursor = self.conn.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        return cursor.fetchall()

    def get_all_sessions(self):
        cursor = self.conn.execute("SELECT DISTINCT session_id FROM messages ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]

db = DatabaseManager()

# --- 3. VECTOR MANAGER (The 429 & Token Solution) ---
class VectorManager:
    def __init__(self):
        # We use a local model for embeddings to save your Google API Quota
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def build_index(self, uploads, local_folder="documents"):
        texts = []
        # 1. Read local documents folder
        if os.path.exists(local_folder):
            for f in os.listdir(local_folder):
                texts.append(self._extract(os.path.join(local_folder, f), is_path=True))
        # 2. Read user uploads
        if uploads:
            for f in uploads:
                texts.append(self._extract(f))
        
        combined = "\n".join(filter(None, texts))
        if not combined.strip(): return None

        # 3. Chunk and Store
        chunks = self.text_splitter.split_text(combined)
        return FAISS.from_texts(chunks, self.embeddings)

    def _extract(self, source, is_path=False):
        try:
            name = source if is_path else source.name
            ext = os.path.splitext(name)[-1].lower()
            if ext == ".pdf":
                return "\n".join([p.extract_text() for p in PdfReader(source).pages if p.extract_text()])
            elif ext == ".docx":
                import docx2txt
                return docx2txt.process(source)
            elif ext == ".txt":
                if is_path:
                    with open(source, "r", encoding="utf-8", errors="ignore") as f: return f.read()
                return source.getvalue().decode("utf-8", errors="ignore")
        except: return ""

vm = VectorManager()

# --- 4. STREAMLIT UI ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

with st.sidebar:
    st.title("Settings")
    if st.button("➕ New Chat"):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.rerun()
    
    sessions = db.get_all_sessions()
    if sessions:
        selected = st.selectbox("Chat History", sessions)
        if selected != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected
            st.rerun()

    uploads = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 AI Research Assistant")

# Display Messages
for role, content in db.get_history(st.session_state.current_chat_id):
    with st.chat_message(role): st.markdown(content)

# Input
user_input = st.chat_input("Ask about your data...")

if user_input:
    db.save_message(st.session_state.current_chat_id, "user", user_input)
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # 1. Search for context (Prevents sending massive text to API)
        with st.spinner("Searching documents..."):
            vector_db = vm.build_index(uploads)
            context = ""
            if vector_db:
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n\n".join([d.page_content for d in docs])

        # 2. Call Gemini with Retry Logic (Handles 429 Errors)
        full_res = ""
        success = False
        for attempt in range(3):
            try:
                prompt = f"Use this CONTEXT to answer:\n{context}\n\nUSER QUESTION: {user_input}"
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                for chunk in response:
                    if chunk.text:
                        full_res += chunk.text
                        placeholder.markdown(full_res + "▌")
                success = True
                break 
            except exceptions.ResourceExhausted:
                wait_time = (attempt + 1) * 15
                placeholder.warning(f"Quota exceeded. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not success:
            st.error("API is overloaded. Please try again in 1 minute.")
        else:
            placeholder.markdown(full_res)
            db.save_message(st.session_state.current_chat_id, "assistant", full_res)
