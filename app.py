import streamlit as st
import os
import time
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

load_dotenv()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Resilient Research Bot", layout="wide")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Check your Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. THE ERROR ERADICATOR (Retry Logic) ---
# This decorator will automatically retry the function if a 429 error occurs.
# It waits exponentially (1s, 2s, 4s...) and adds 'jitter' to prevent collisions.
@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_gemini_api(prompt, context):
    # SWITCHED MODEL: 'gemini-1.5-flash' is more stable for Free Tier than 2.0
    return client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"CONTEXT:\n{context[:15000]}\n\nQUESTION: {prompt}"
    )

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

# --- 4. UI & LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Pro Controls")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    uploads = st.file_uploader("Upload Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context_text = ""
        if os.path.exists("documents"):
            for f in os.listdir("documents"):
                context_text += extract_text(os.path.join("documents", f), is_path=True) + "\n"
        if uploads:
            for f in uploads: context_text += extract_text(f) + "\n"
        
        try:
            # We use the retry-wrapped function here
            response = call_gemini_api(prompt, context_text)
            
            full_res = response.text
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
            
        except Exception as e:
            if "429" in str(e):
                st.error("🚨 Even after retries, the quota is exhausted. Please wait 1-2 minutes.")
            else:
                st.error(f"Error: {e}")
