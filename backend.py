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
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        # Added session_id to separate different users
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
            st.error("API Key not found!")
            st.stop()
        
        # Upgraded to Gemini 2.5 Flash for 2026 stability
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            streaming=True,
            temperature=0,
            api_version="v1",
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
        self.db = DatabaseManager()
        self.knowledge_base = self._load_backend_documents("documents")

    def _load_backend_documents(self, folder_path):
        combined_text = ""
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
                        if text: combined_text += text + " "
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        combined_text += f.read() + " "
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
        
        return " ".join(combined_text.split())[:15000]

    def get_streaming_response(self, user_input, session_id):
        # Retrieve history ONLY for this specific session
        raw_history = self.db.get_session_history(session_id)
        
        context_prompt = f"INTERNAL DOCUMENTS CONTEXT:\n{self.knowledge_base}\n\n"
        
        messages = [
            SystemMessage(content="You are a professional assistant. Use the context to answer accurately.")
        ]
        
        # Add the last 5 private messages from this specific user
        for role, content in raw_history[-5:]:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        
        messages.append(HumanMessage(content=f"{context_prompt}USER QUESTION: {user_input}"))
        return self.llm.stream(messages)
