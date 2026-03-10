import streamlit as st
import os
import sqlite3
import uuid
import time
import docx2txt
from google import genai
from google.genai import errors
from pypdf import PdfReader
from dotenv import load_dotenv

# Modern LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key, vertexai=False)

client = get_client()

# --- 2. DATABASE MANAGER (With Migration Support) ---
class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()
        self.migrate_schema() # Automatically fix missing columns

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, role TEXT, content TEXT,
                prompt_tokens INTEGER DEFAULT 0, 
                completion_tokens INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def migrate_schema(self):
        """Adds missing token columns to existing databases."""
        cursor = self.conn.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "prompt_tokens" not in columns:
            self.conn.execute("ALTER TABLE messages ADD COLUMN prompt_tokens INTEGER DEFAULT 0")
        if "completion_tokens" not in columns:
            self.conn.execute("ALTER TABLE messages ADD COLUMN completion_tokens INTEGER DEFAULT 0")
        self.conn.commit()

    def save_message(self, session_id, role, content, p_tokens=0, c_tokens=0):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, p_tokens, c_tokens)
        )
        self.conn.commit()

    def get_history(self, session_id):
        cursor = self.conn.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        return cursor.fetchall()

    def get_total_usage(self):
        cursor = self.conn.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens), COUNT(id) FROM messages")
        row = cursor.fetchone()
        return (row[0] or 0, row[1] or 0, row[2] or 0)

db = DatabaseManager()

# --- 3. VECTOR MANAGER ---
class VectorManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def build_index(self, uploads):
        texts = []
        if uploads:
            for f in uploads:
                ext = os.path.splitext(f.name)[-1].lower()
                try:
                    if ext == ".pdf":
                        texts.append("\n".join([p.extract_text() for p in PdfReader(f).pages if p.extract_text()]))
                    elif ext == ".docx":
                        texts.append(docx2txt.process(f))
                    elif ext == ".txt":
                        texts.append(f.getvalue().decode("utf-8", errors="ignore"))
                except Exception as e:
                    st.warning(f"Error reading {f.name}: {e}")
        
        combined = "\n".join(filter(None, texts))
        if not combined.strip(): return None
        chunks = self.text_splitter.split_text(combined)
        return FAISS.from_texts(chunks, self.embeddings)

vm = VectorManager()

# --- 4. SIDEBAR DASHBOARD ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

with st.sidebar:
    st.title("📊 Usage Dashboard")
    p_sum, c_sum, msg_count = db.get_total_usage()
    
    col1, col2 = st.columns(2)
    col1.metric("Input Tokens", f"{p_sum:,}")
    col2.metric("Output Tokens", f"{c_sum:,}")
    
    daily_limit = 1500
    remaining = max(0, daily_limit - msg_count)
    st.progress(remaining / daily_limit, text=f"Daily Quota: {remaining}/{daily_limit} left")
    
    st.divider()
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4()); st.rerun()

    uploads = st.file_uploader("Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 5. MAIN CHAT ---
st.title("🤖 AI Research Assistant")

for role, content in db.get_history(st.session_state.current_chat_id):
    with st.chat_message(role): st.markdown(content)

user_input = st.chat_input("Ask a question...")

if user_input:
    db.save_message(st.session_state.current_chat_id, "user", user_input)
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("Searching documents..."):
            vector_db = vm.build_index(uploads)
            context = ""
            if vector_db:
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n\n".join([d.page_content for d in docs])
    
        full_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=full_prompt
            )
            
            full_res = response.text
            placeholder.markdown(full_res)
            
            db.save_message(
                st.session_state.current_chat_id, 
                "assistant", 
                full_res,
                p_tokens=response.usage_metadata.prompt_token_count,
                c_tokens=response.usage_metadata.candidates_token_count
            )
            st.rerun()
            
        except errors.ClientError as ce:
            if "429" in str(ce):
                st.error("Quota Exceeded! Please wait 60 seconds.")
            else:
                st.error(f"API Error: {ce}")
