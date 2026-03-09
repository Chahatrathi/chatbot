 import sqlite3
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        # Using check_same_thread=False is correct for Streamlit's multi-threaded nature
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

    def get_session_history(self, session_id):
        cursor = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", 
            (session_id,)
        )
        return cursor.fetchall()

class ChatBackend:
    def __init__(self):
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("API Key not found! Please set GOOGLE_API_KEY in .env or secrets.")
            st.stop()
        
        # Use a verified model name like 'gemini-1.5-flash' or 'gemini-2.0-flash-exp'
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            streaming=True,
            temperature=0,
        )
        self.db = DatabaseManager()
        # Load documents on initialization
        self.knowledge_base = self._load_backend_documents("documents")

    def _load_backend_documents(self, folder_path):
        combined_text = []
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return ""

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text: combined_text.append(text)
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        combined_text.append(f.read())
            except Exception as e:
                st.warning(f"Skipped {filename} due to error: {e}")
        
        # Join list into a single string
        final_text = "\n".join(combined_text)
        # Gemini handles large context well; 30k-50k chars is usually safe for basic prompts
        return final_text[:50000] 

    def get_streaming_response(self, user_input, session_id):
        raw_history = self.db.get_session_history(session_id)
        
        # System prompt defines the persona and provides the "Knowledge Base"
        messages = [
            SystemMessage(content=(
                "You are a professional assistant. "
                "Below is the content of internal documents. Use this as your primary source of truth:\n\n"
                f"{self.knowledge_base}"
            ))
        ]
        
        # Add historical context (sliding window of last 5 messages)
        for role, content in raw_history[-5:]:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        
        # Add the current user question
        messages.append(HumanMessage(content=user_input))
        
        return self.llm.stream(messages)
