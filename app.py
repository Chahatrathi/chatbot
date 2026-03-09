import streamlit as st
import os
import time
import uuid
import docx
from google import genai
from google.genai import errors
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETTINGS & RE-INITIALIZATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# This ensures that if you change the key in the UI or secrets, the client updates
def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to .env or Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DOCUMENT UTILITIES ---

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

def load_all_context(uploads):
    context = ""
    # 1. Load from documents folder
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    # 2. Load from manual uploads
    if uploads:
        for f in uploads: context += extract_text(f) + "\n"
    
    # CRITICAL: Free tier has a low "Input Tokens Per Minute" limit.
    # We limit this to ~10,000 characters to prevent 429 errors.
    return context[:10000] 

# --- 3. THE CHAT ENGINE (With Smart Retry) ---

def ask_gemini(prompt):
    # If the first attempt fails, we wait and retry with a cleaner state
    for attempt in range(3):
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt
            )
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(attempt * 5 + 5) # Wait 5s, then 10s
                continue
            raise e

# --- 4. SESSION & UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Assistant Control")
    if st.button("🗑️ Clear Chat & Cache"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Upload more data", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context = load_all_context(uploads)
        full_query = f"Context: {context}\n\nUser Question: {prompt}\nAnswer using the context above."
        
        try:
            placeholder = st.empty()
            full_res = ""
            for chunk in ask_gemini(full_query):
                full_res += chunk.text
                placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
        except Exception as e:
            st.error(f"Error: {e}. Try the 'Clear Chat & Cache' button.")
