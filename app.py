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
