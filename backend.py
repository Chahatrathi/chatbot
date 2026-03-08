import sqlite3
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_message(self, role, content):
        self.conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        self.conn.commit()

    def get_all_history(self):
        cursor = self.conn.execute("SELECT role, content FROM messages ORDER BY timestamp ASC")
        return cursor.fetchall()

class ChatBackend:
    def __init__(self):
        api_key = st.secrets["GOOGLE_API_KEY"]
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, streaming=True)
        self.db = DatabaseManager()
        # Load all documents from the backend folder on initialization
        self.knowledge_base = self._load_backend_documents("documents")

    def _load_backend_documents(self, folder_path):
        """Scans the backend folder and extracts text from all supported files."""
        combined_text = ""
        if not os.path.exists(folder_path):
            return "No documents found in the backend."

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        combined_text += page.extract_text() + "\n"
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        combined_text += f.read() + "\n"
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return combined_text

    def get_streaming_response(self, user_input):
        raw_history = self.db.get_all_history()
        
        # System instructions including the backend knowledge
        sys_prompt = (
            "You are an expert assistant. You have access to the following document data "
            "stored in the backend. Use it to provide accurate answers:\n\n"
            f"{self.knowledge_base}"
        )
        
        messages = [SystemMessage(content=sys_prompt)]
        
        for role, content in raw_history:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        
        messages.append(HumanMessage(content=user_input))
        return self.llm.stream(messages)