import streamlit as st
import os
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gemini Pro Assistant", layout="wide")

def get_client():
    # Ensure your API key is from a project with BILLING ENABLED for Pro limits
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check your Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DOCUMENT HANDLING ---
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

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Assistant Admin")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Upload Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- 4. CHAT INTERFACE ---
st.title("🤖 Assistant Research Chatbot")

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Logic for Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # Load Context from 'documents/' folder and manual uploads
        context = ""
        if os.path.exists("documents"):
            for f in os.listdir("documents"):
                context += extract_text(os.path.join("documents", f), is_path=True) + "\n"
        if uploads:
            for f in uploads: context += extract_text(f) + "\n"
        
        try:
            placeholder = st.empty()
            full_res = ""
            
            # CRITICAL FIX: Use the model name ONLY (no "models/" prefix)
            # Use "gemini-2.0-flash" for the best 2026 performance
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash", 
                contents=f"CONTEXT:\n{context[:60000]}\n\nQUESTION: {prompt}"
            )
            
            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    placeholder.markdown(full_res + "▌")
            
            placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            # If 2.0-flash isn't available for your key, try 1.5-flash as fallback
            st.error(f"Execution Error: {e}")
            st.info("💡 Pro Tip: If 'NOT_FOUND' persists, ensure your project has the 'Generative Language API' enabled in Google Cloud Console.")
