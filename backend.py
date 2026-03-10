import sqlite3
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

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
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            streaming=True,
            temperature=0,
        )

        # Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.db = DatabaseManager()
        self.vector_store = self._initialize_vector_store("documents")

    def _initialize_vector_store(self, folder_path):
        """Processes documents into chunks and stores them in FAISS."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return None

        all_text = ""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        all_text += page.extract_text() or ""
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        all_text += f.read()
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")

        if not all_text.strip():
            return None

        # 1. Chunking: Split text into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(all_text)

        # 2. Vector Store: Create FAISS index using embeddings (Cosine Similarity)
        # FAISS uses L2 distance by default, but for normalized embeddings, it is equivalent to Cosine.
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        return vector_store

    def get_streaming_response(self, user_input, session_id):
        # 1. Similarity Search: Find top 3 relevant chunks
        context = ""
        if self.vector_store:
            docs = self.vector_store.similarity_search(user_input, k=3)
            context = "\n---\n".join([doc.page_content for doc in docs])

        # 2. Build Messages
        messages = [
            SystemMessage(content=(
                "You are a professional assistant. Use the following context to answer the user. "
                "If the answer isn't in the context, use your knowledge but state it's not in the docs.\n\n"
                f"CONTEXT:\n{context}"
            ))
        ]
        
        # Add history (sliding window)
        raw_history = self.db.get_session_history(session_id)
        for role, content in raw_history[-5:]:
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        
        messages.append(HumanMessage(content=user_input))
        
        return self.llm.stream(messages)
