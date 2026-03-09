import streamlit as st
import os
import time
import docx
from google import genai
from google.genai import types, errors
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Gemini Pro Assistant", layout="wide")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add to .env or Streamlit Secrets.")
        st.stop()
    # Initializing with the latest v1 stable API
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1"))

client = get_client()

# --- 2. ROBUST RETRY LOGIC ---
def generate_with_backoff(model_id, contents, retries=3):
    """Retries the request with increasing delays if a 429 occurs."""
    for i in range(retries):
        try:
            return client.models.generate_content_stream(
                model=model_id,
                contents=contents
            )
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                wait_time = (i + 1) * 10  # 10s, 20s...
                st.warning(f"Quota reached. Sleeping {wait_time}s before retry...")
                time.sleep(wait_time)
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

# --- 4. UI & CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Pro Controls")
    if st.button("🗑️ Reset All"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    uploads = st.file_uploader("Upload Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # Compile local knowledge
        context = ""
        if os.path.exists("documents"):
            for f in os.listdir("documents"):
                context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
        if uploads:
            for f in uploads: context += extract_text(f) + "\n"
        
        try:
            placeholder = st.empty()
            full_res = ""
            
            # Using 'gemini-3-flash-preview' for 2026 performance (or 'gemini-2.0-flash')
            # Ensure model ID is just the string, no "models/" prefix.
            stream = generate_with_backoff(
                model_id="gemini-3-flash-preview", 
                contents=f"CONTEXT:\n{context[:40000]}\n\nQUESTION: {prompt}"
            )
            
            for chunk in stream:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Error: {e}")
