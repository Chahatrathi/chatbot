import sqlite3
import streamlit as st
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

    def get_chat_as_text(self):
        history = self.get_all_history()
        return "\n".join([f"[{r.upper()}]: {c}\n{'-'*20}" for r, c in history]) if history else "No history."

class ChatBackend:
    def __init__(self):
        # Automatically pull key from Streamlit Secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            streaming=True
        )
        self.db = DatabaseManager()

    def get_streaming_response(self, user_input):
        raw_history = self.db.get_all_history()
        messages = [SystemMessage(content="You are a helpful assistant.")]
        
        for role, content in raw_history:
            msg_type = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
            messages.append(msg_type)
        
        messages.append(HumanMessage(content=user_input))
        return self.llm.stream(messages)
