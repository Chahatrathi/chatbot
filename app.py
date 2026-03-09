import streamlit as st
import os
import time
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Pro Research Bot", layout="wide")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check Streamlit Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DATA EXTRACTION ---
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

def load_context(uploads):
    context = ""
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    if uploads:
        for f in uploads: context += extract_text(f) + "\n"
    # TPM Management: Truncate to save tokens and prevent quota hits
    return context[:30000] 

# --- 3. THE SAFE GENERATOR (Retries on 429) ---
def safe_generate(prompt, context):
    """Intercepts 429 errors and retries with increasing delays."""
    full_query = f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"
    
    for attempt in range(5): # Up to 5 retries
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=full_query
            )
        except Exception as e:
            if "429" in str(e):
                # Exponential delay: 8s, 16s, 24s...
                wait_time = (attempt + 1) * 8 
                st.warning(f"Quota reached. Auto-retrying in {wait_time}s... (Attempt {attempt+1}/5)")
                time.sleep(wait_time)
                continue
            raise e
    st.error("Maximum retries reached. Please wait 60 seconds for the quota window to reset.")
    return None

# --- 4. UI & CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Admin")
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    uploads = st.file_uploader("Upload Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask about equity..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        context_data = load_context(uploads)
        
        try:
            stream = safe_generate(prompt, context_data)
            if stream:
                for chunk in stream:
                    if chunk.text:
                        full_res += chunk.text
                        placeholder.markdown(full_res + "▌")
                placeholder.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
        except Exception as e:
            st.error(f"Error: {e}")
