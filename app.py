import streamlit as st
import os
import docx
import uuid
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. PRO CONFIGURATION ---
st.set_page_config(page_title="Gemini Pro Assistant", layout="wide")

def get_pro_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add to Streamlit Secrets or .env")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_pro_client()

# --- 2. DOCUMENT UTILITIES ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        if ext := os.path.splitext(file_source if is_path else file_source.name)[-1].lower():
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

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Assistant Control")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    uploads = st.file_uploader("Upload Docs", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 4. MAIN INTERFACE ---
st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("What is equity?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # Gather local context
        context = ""
        if os.path.exists("documents"):
            for f in os.listdir("documents"):
                context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
        if uploads:
            for f in uploads: context += extract_text(f) + "\n"
        
        try:
            placeholder = st.empty()
            full_res = ""
            
            # Use 'gemini-1.5-flash' or 'gemini-2.0-flash' for maximum compatibility
            # DO NOT use 'models/' prefix here.
            response = client.models.generate_content_stream(
                model="gemini-1.5-flash", 
                contents=f"Context: {context[:50000]}\n\nQuestion: {prompt}"
            )
            
            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.info("💡 Tip: Ensure you are using 'gemini-1.5-flash' or 'gemini-2.0-flash' in the code.")
