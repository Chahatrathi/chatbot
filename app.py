import streamlit as st
import os
import time
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Pro Research Bot", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check your Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. EXPONENTIAL BACKOFF LOGIC ---
def generate_response_with_retry(prompt_text, uploads, retries=3):
    """Eradicates 429 errors by catching them and retrying after a delay."""
    
    # Compile context
    context = ""
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    if uploads:
        for f in uploads:
            context += extract_text(f) + "\n"
    
    # Keep context within reasonable limits to save tokens
    full_query = f"CONTEXT:\n{context[:30000]}\n\nQUESTION: {prompt_text}"
    
    delay = 5  # Initial wait time in seconds
    for i in range(retries):
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=full_query
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                if i < retries - 1:
                    st.warning(f"Quota exceeded. Retrying in {delay} seconds... (Attempt {i+1})")
                    time.sleep(delay)
                    delay *= 2  # Double the wait time for the next retry
                    continue
            raise e

# --- 3. DATA EXTRACTION ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        ext = os.path.splitext(file_source if is_path else file_source.name)[-1].lower()
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

# --- 4. UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Pro Controls")
    if st.button("🗑️ Clear Chat & Cache"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    uploads = st.file_uploader("Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            full_res = ""
            
            # Use the retry-wrapped function
            stream = generate_response_with_retry(user_input, uploads)
            
            for chunk in stream:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
