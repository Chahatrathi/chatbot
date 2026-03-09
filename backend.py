import sqlite3
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load local .env file if it exists (for VS Code development)
load_dotenv()

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
        # 1. Try Streamlit Secrets (Production) 2. Try .env (Local)
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("API Key not found! Check your Secrets or .env file.")
            st.stop()
        
        # KEY FIX: Explicitly forcing 'v1' API version to avoid the 404 error
        # Some versions of LangChain use 'api_version', others use 'version'.
        # We provide both to ensure compatibility.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            streaming=True,
            temperature=0,
            api_version="v1",  # Primary fix
            model_kwargs={"api_version": "v1"}, # Fallback fix for newer SDKs
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
        
        # Clean text and limit size to ensure stability
        cleaned_text = " ".join(combined_text.split())
        return cleaned_text[:15000]

    def get_streaming_response(self, user_input):
        raw_history = self.db.get_all_history()
        
        # Combine context and question into a single human prompt
        # This is more stable than using a heavy SystemMessage
        context_prompt = f"INTERNAL DOCUMENTS CONTEXT:\n{self.knowledge_base}\n\n"
        
        messages = [
            SystemMessage(content="You are a helpful assistant. Use the provided context to answer questions accurately.")
        ]
        
        # Include last 3 exchanges for context
        for role, content in raw_history[-3:]:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        
        messages.append(HumanMessage(content=f"{context_prompt}USER QUESTION: {user_input}"))
        
        return self.llm.stream(messages)
