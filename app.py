import streamlit as st
import os
import sqlite3
import uuid
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

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

# --- 2. DATABASE MANAGER ---
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

# --- 3. DATA EXTRACTION ---
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

# --- 4. SESSION MANAGEMENT ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

def start_new_chat():
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.rerun()

# --- 5. SIDEBAR (History & Files) ---
with st.sidebar:
    st.title("Research Control")
    if st.button("➕ Start New Chat", use_container_width=True):
        start_new_chat()
    
    st.divider()
    
    # Feature: List Old Sessions
    st.subheader("Previous Chats")
    sessions = db.get_all_sessions()
    if sessions:
        # User selects an old session to load it
        selected_session = st.selectbox(
            "Select a chat to view:",
            sessions,
            index=sessions.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in sessions else 0
        )
        if selected_session != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_session
            st.rerun()
            
        # Download Button for the selected session
        history_data = db.get_history(st.session_state.current_chat_id)
        if history_data:
            chat_text = "\n".join([f"{r.upper()}: {c}" for r, c in history_data])
            st.download_button(
                label="📥 Download This Chat",
                data=chat_text,
                file_name=f"chat_{st.session_state.current_chat_id[:8]}.txt",
                use_container_width=True
            )
    else:
        st.info("No chat history found.")

    st.divider()
    uploads = st.file_uploader("Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 6. MAIN INTERFACE ---
st.title("🤖 Assistant Research Chatbot")

# Display Messages
for role, content in db.get_history(st.session_state.current_chat_id):
    with st.chat_message(role):
        st.markdown(content)

# CHAT INPUT
user_input = st.chat_input("Ask a question...")

if user_input:
    # Save and Display User Message
    db.save_message(st.session_state.current_chat_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        # Compile Context
        context = ""
        if os.path.exists("documents"):
            for f in os.listdir("documents"):
                context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
        if uploads:
            for f in uploads:
                context += extract_text(f) + "\n"

        # System Instruction: Fallback to General Knowledge
        prompt_template = (
            "You are a helpful research assistant. "
            "Use the provided CONTEXT to answer the user's question. "
            "If the CONTEXT is missing or does not contain the answer, answer the question using your general knowledge."
            "\n\n"
            f"CONTEXT:\n{context[:30000] if context else 'No document context provided.'}\n\n"
            f"QUESTION: {user_input}"
        )

        try:
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt_template
            )
            
            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            db.save_message(st.session_state.current_chat_id, "assistant", full_res)
            
        except Exception as e:
            if "429" in str(e):
                st.error("Rate limit hit. Please wait a moment.")
            else:
                st.error(f"Error: {e}")
