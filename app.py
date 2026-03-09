import streamlit as st
import os
import time
import uuid
import docx
from google import genai
from google.genai import errors
from pypdf import PdfReader
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

# API Key Retrieval
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key not found! Add it to Streamlit Secrets or a .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. DOCUMENT PROCESSING ---

@st.cache_data(show_spinner="Analyzing documents...")
def extract_text(file_source, is_path=False):
    """Extracts text from PDF, DOCX, or TXT."""
    try:
        name = file_source if is_path else file_source.name
        ext = os.path.splitext(name)[-1].lower()
        text = ""

        if ext == ".pdf":
            reader = PdfReader(file_source)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == ".docx":
            doc = docx.Document(file_source)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = file_source.getvalue().decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        return f"Error reading {name}: {str(e)}"

def get_context_from_folder(folder="documents"):
    """Reads all files in the documents/ folder."""
    context = ""
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                context += extract_text(path, is_path=True) + "\n"
    return context

# --- 3. ERROR HANDLING & RETRY LOGIC ---

def generate_response_with_retry(prompt, retries=3, delay=5):
    """Handles 429 errors by waiting and retrying."""
    for i in range(retries):
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt
            )
        except errors.ClientError as e:
            if "429" in str(e):
                if i < retries - 1:
                    st.warning(f"Quota reached. Retrying in {delay}s... (Attempt {i+1}/{retries})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
            raise e

# --- 4. SESSION MANAGEMENT ---
if "all_chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.all_chats = {cid: {"name": "New Chat", "messages": []}}
    st.session_state.current_chat_id = cid

def start_new_chat():
    cid = str(uuid.uuid4())
    st.session_state.all_chats[cid] = {"name": "New Chat", "messages": []}
    st.session_state.current_chat_id = cid
    st.rerun()

# --- 5. UI & SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
    
    chat_ids = list(st.session_state.all_chats.keys())
    selected_id = st.selectbox(
        "History", 
        options=chat_ids, 
        format_func=lambda x: st.session_state.all_chats[x]["name"],
        index=chat_ids.index(st.session_state.current_chat_id)
    )
    st.session_state.current_chat_id = selected_id
    
    st.divider()
    uploads = st.file_uploader("Upload Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 6. CHAT INTERFACE ---
st.title("🤖 Assistant Research Chatbot")
active_chat = st.session_state.all_chats[st.session_state.current_chat_id]

# Display history
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
if user_input := st.chat_input("Ask a question about your data..."):
    # Update chat name if first message
    if not
