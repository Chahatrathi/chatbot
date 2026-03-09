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
st.set_page_config(page_title="Gemini Pro Research Bot", layout="wide")

def get_pro_client():
    # Ensure your API key is from a project with BILLING ENABLED
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing!")
        st.stop()
    # The new SDK handles versioning automatically
    return genai.Client(api_key=api_key)

client = get_pro_client()

# --- 2. DOCUMENT PROCESSING ---
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
    except Exception as e:
        return ""
    return ""

def load_context(uploads):
    context = ""
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    if uploads:
        for f in uploads: context += extract_text(f) + "\n"
    return context[:150000] # Pro handles large context easily

# --- 3. UI & CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Pro Settings")
    if st.button("🔄 Clear App Cache"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    uploads = st.file_uploader("Upload Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask about investment..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context_data = load_context(uploads)
        
        try:
            placeholder = st.empty()
            full_response = ""
            
            # Use 'gemini-1.5-pro' or 'gemini-2.0-flash'
            # Note: Do NOT use 'models/gemini-1.5-pro' with this specific SDK
            response = client.models.generate_content_stream(
                model="gemini-1.5-pro", 
                contents=f"CONTEXT:\n{context_data}\n\nQUESTION: {prompt}"
            )
            
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
