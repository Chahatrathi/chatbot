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

# Modern LangChain imports for 2026
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="centered", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key, vertexai=False)

client = get_client()

# --- 2. DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()
        self.migrate_schema()

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

    def get_all_sessions(self):
        cursor = self.conn.execute("SELECT DISTINCT session_id FROM messages ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]

db = DatabaseManager()

# --- 3. VECTOR MANAGER (Optimized RAG) ---
class VectorManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # STICKY RAG: Reduced chunk size to 500 characters
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    @st.cache_resource(show_spinner=False)
    def get_vector_store(_self, folder_path="documents"):
        if not os.path.exists(folder_path):
            return None
        
        texts = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[-1].lower()
            try:
                if ext == ".pdf":
                    texts.append("\n".join([p.extract_text() for p in PdfReader(file_path).pages if p.extract_text()]))
                elif ext == ".docx":
                    texts.append(docx2txt.process(file_path))
                elif ext == ".txt":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        combined = "\n".join(filter(None, texts))
        if not combined.strip(): return None
        chunks = _self.text_splitter.split_text(combined)
        return FAISS.from_texts(chunks, _self.embeddings)

vm = VectorManager()

# --- 4. SIDEBAR (Simplified: New Chat & History) ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

with st.sidebar:
    st.title("💬 Chat Controls")
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.subheader("Previous Conversations")
    sessions = db.get_all_sessions()
    if sessions:
        selected = st.selectbox(
            "Select a chat to view:", 
            sessions, 
            index=sessions.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in sessions else 0
        )
        if selected != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected
            st.rerun()
    else:
        st.info("No chat history found.")

# --- 5. MAIN INTERFACE ---
st.title("🤖 AI Research Assistant")

# Display Messages from selected history
for role, content in db.get_history(st.session_state.current_chat_id):
    with st.chat_message(role):
        st.markdown(content)

# User Query
user_input = st.chat_input("Ask a question about your local documents...")

if user_input:
    # 1. Save and Display User Input
    db.save_message(st.session_state.current_chat_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # SEARCH FOR CONTEXT (MANDATORY RAG)
        with st.spinner("Retrieving relevant context..."):
            vector_db = vm.get_vector_store()
            context = ""
            if vector_db:
                # Retrieve only top 3 small chunks (approx 1500 chars total)
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n\n".join([d.page_content for d in docs])
    
        # Use Context to answer
        prompt = f"Use this CONTEXT to answer the question briefly:\n{context}\n\nQUESTION: {user_input}"
        
        # API CALL WITH RETRY LOGIC
        success = False
        full_res = ""
        for attempt in range(3):
            try:
                # Using 3.1 Flash-Lite (the March 2026 standard for high-quota free tier)
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=prompt
                )
                
                full_res = response.text
                placeholder.markdown(full_res)
                
                # Save message with metadata
                db.save_message(
                    st.session_state.current_chat_id, 
                    "assistant", 
                    full_res,
                    p_tokens=response.usage_metadata.prompt_token_count,
                    c_tokens=response.usage_metadata.candidates_token_count
                )
                success = True
                break
                
            except errors.ClientError as ce:
                if "429" in str(ce):
                    placeholder.warning(f"Rate limit hit. Retrying in 30s... (Attempt {attempt+1}/3)")
                    time.sleep(30)
                else:
                    st.error(f"API Error: {ce}")
                    break

        if success:
            st.rerun() # Refresh to update chat display properly
