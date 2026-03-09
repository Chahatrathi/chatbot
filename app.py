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

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Gemini Pro Research Bot", layout="wide")

# For Paid/Pro users, ensure your Google Cloud Project has billing enabled.
# You can pass the 'project' ID if using Vertex AI, otherwise API Key works for AI Studio Pro.
def get_pro_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check your Streamlit Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_pro_client()

# --- 2. DOCUMENT PROCESSING ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        ext = os.path.splitext(file_source if is_path else file_source.name)[-1].lower()
        text = ""
        if ext == ".pdf":
            reader = PdfReader(file_source)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == ".docx":
            doc = docx.Document(file_source)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
            else:
                text = file_source.getvalue().decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        return f"Error: {e}"

def load_context(uploads):
    context = ""
    if os.path.exists("documents"):
        for f in os.listdir("documents"):
            context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
    if uploads:
        for f in uploads: context += extract_text(f) + "\n"
    # Pro users can handle much larger context (up to 2M tokens), 
    # but we'll cap at 100k chars for speed.
    return context[:100000]

# --- 3. PRO-LEVEL CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Pro Assistant Settings")
    if st.button("🔄 Hard Reset App"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.info("Status: Gemini Pro Tier Active")
    uploads = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display Messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context_data = load_context(uploads)
        
        # System instructions help minimize hallucinations
        system_instruction = "You are a senior financial researcher. Use the provided context to answer. Be precise."
        
        try:
            placeholder = st.empty()
            full_response = ""
            
            # Using Gemini 1.5 Pro or 2.0 Flash (Paid tier handles both with high RPM)
            response = client.models.generate_content_stream(
                model="gemini-1.5-pro", # Use 'gemini-1.5-pro' for complex reasoning
                contents=f"CONTEXT:\n{context_data}\n\nQUESTION: {prompt}"
            )
            
            for chunk in response:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except errors.ClientError as e:
            if "429" in str(e):
                st.error("🚨 Quota Error: Even with Pro, you might be hitting a per-minute limit. Wait 10 seconds and try again.")
            else:
                st.error(f"Error: {e}")
