import streamlit as st
import os
import time
import uuid
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Assistant Research Chatbot", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to Streamlit Secrets or a .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. THE ERROR ERADICATOR (Safe Generation) ---
def safe_generate_stream(prompt_text, context_data, retries=3):
    """Handles 429 errors by waiting and retrying with exponential backoff."""
    # TPM Management: Truncate context to stay under free/pro per-minute token limits
    # 30,000 characters is roughly 8k-10k tokens, safe for most tiers.
    safe_context = context_data[:30000]
    full_prompt = f"CONTEXT:\n{safe_context}\n\nUSER QUESTION: {prompt_text}"
    
    delay = 10 # Initial wait in seconds for a 429 error
    
    for attempt in range(retries):
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=full_prompt
            )
        except Exception as e:
            if "429" in str(e):
                if attempt < retries - 1:
                    st.warning(f"Quota hit (429). Cooling down for {delay}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                    delay *= 2 # Exponentially increase wait time
                    continue
            raise e

# --- 3. DATA EXTRACTION UTILITIES ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        name = file_source if is_path else file_source.name
        ext = os.path.splitext(name)[-1].lower()
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

def load_all_docs(uploaded_files):
    context = ""
    # Load from local 'documents' folder
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    # Load from current session uploads
    if uploaded_files:
        for f in uploaded_files:
            context += extract_text(f) + "\n"
    return context

# --- 4. SESSION MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. UI LAYOUT ---
with st.sidebar:
    st.title("Pro Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Upload Additional Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display historical messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
if user_input := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            full_response = ""
            
            # Gather all text data
            combined_context = load_all_docs(uploads)
            
            # Stream the response using our retry-safe function
            stream = safe_generate_stream(user_input, combined_context)
            
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.info("💡 If 429 persists, please wait 60 seconds for the API window to reset.")
