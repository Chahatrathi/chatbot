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

# --- 1. CONFIGURATION & CLIENT FIX ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    # Priority: Streamlit Secrets -> .env file
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("🔑 API Key missing! Add it to App Settings > Secrets.")
        st.stop()
        
    try:
        # vertexai=False is the SPECIFIC fix for the ClientError
        # it forces the SDK to use the standard API Studio key path
        return genai.Client(api_key=api_key, vertexai=False)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        st.stop()

client = get_client()

# --- 2. DATABASE MANAGER ---
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

# --- 3. VECTOR MANAGER (Optimizes Context) ---
class VectorManager:
    def __init__(self):
        # Local model: Saves API quota for actual chat
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def build_index(self, uploads):
        texts = []
        if uploads:
            for f in uploads:
                name = f.name
                ext = os.path.splitext(name)[-1].lower()
                try:
                    if ext == ".pdf":
                        texts.append("\n".join([p.extract_text() for p in PdfReader(f).pages if p.extract_text()]))
                    elif ext == ".docx":
                        texts.append(docx2txt.process(f))
                    elif ext == ".txt":
                        texts.append(f.getvalue().decode("utf-8", errors="ignore"))
                except Exception as e:
                    st.warning(f"Error reading {name}: {e}")
        
        combined = "\n".join(filter(None, texts))
        if not combined.strip(): return None

        chunks = self.text_splitter.split_text(combined)
        return FAISS.from_texts(chunks, self.embeddings)

vm = VectorManager()

# --- 4. STREAMLIT UI ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

with st.sidebar:
    st.title("Research Control")
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4()); st.rerun()
    
    sessions = db.get_all_sessions()
    if sessions:
        selected = st.selectbox("Chat History", sessions)
        if selected != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected; st.rerun()

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
        
        # 1. Search for context
        with st.spinner("Searching documents..."):
            vector_db = vm.build_index(uploads)
            context = ""
            if vector_db:
                docs = vector_db.similarity_search(user_input, k=3)
                context = "\n\n".join([d.page_content for d in docs])

        # 2. Call Gemini with Retry & Explicit Error Handling
        full_res = ""
        success = False
        # Switch to flash-lite if hitting 429 often
        model_name = "gemini-2.0-flash" 

        for attempt in range(3):
            try:
                prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"
                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt
                )
                for chunk in response:
                    if chunk.text:
                        full_res += chunk.text
                        placeholder.markdown(full_res + "▌")
                success = True
                break 
            except errors.ClientError as ce:
                # This catches the specific error you saw and explains it
                st.error(f"API Configuration Error: {ce}")
                break
            except Exception as e:
                if "429" in str(e):
                    wait_time = (attempt + 1) * 20
                    placeholder.warning(f"Quota exceeded. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Unexpected Error: {e}")
                    break
        
        if success:
            placeholder.markdown(full_res)
            db.save_message(st.session_state.current_chat_id, "assistant", full_res)
