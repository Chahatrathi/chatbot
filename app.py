import streamlit as st
import os
import time
import docx
from google import genai
from google.genai import types
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Gemini Pro Assistant", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check your Secrets or .env file.")
        st.stop()
    # Using v1 for stability; the SDK handles the rest
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. FAIL-SAFE GENERATION ---
def generate_response(prompt_text, uploads):
    # Try the fastest stable model first, then fall back
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
    
    # Compile context
    context = ""
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    if uploads:
        for f in uploads:
            context += extract_text(f) + "\n"
    
    full_query = f"CONTEXT:\n{context[:50000]}\n\nQUESTION: {prompt_text}"

    for model_id in models_to_try:
        try:
            return client.models.generate_content_stream(
                model=model_id,
                contents=full_query
            )
        except Exception as e:
            if "404" in str(e):
                continue # Try the next model in the list
            raise e
    raise Exception("No supported models found. Check your API project settings.")

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
    st.title("Pro Settings")
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            full_res = ""
            
            stream = generate_response(user_input, uploads)
            
            for chunk in stream:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Execution Error: {e
