import streamlit as st
import os
import time
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant Pro", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to Streamlit Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DATA EXTRACTION UTILITIES ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        name = file_source if is_path else file_source.name
        ext = os.path.splitext(name)[-1].lower()
        if ext == ".pdf":
            reader = PdfReader(file_source)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == ".docx":
            doc = docx.Document(file_source)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f: return f.read()
            return file_source.getvalue().decode("utf-8", errors="ignore")
    except: return ""
    return ""

def load_knowledge_base(uploads):
    full_text = ""
    # Load from 'documents/' folder automatically
    if os.path.exists("documents"):
        for filename in os.listdir("documents"):
            path = os.path.join("documents", filename)
            if os.path.isfile(path):
                full_text += extract_text(path, is_path=True) + "\n"
    # Load from manual uploader
    if uploads:
        for f in uploads:
            full_text += extract_text(f) + "\n"
    return full_text

# --- 3. THE ERROR ERADICATOR (Logic for 429) ---
def safe_generate(prompt, context_data):
    """Intercepts 429 errors and waits for the quota to reset."""
    # TPM Management: Truncate context to ~30k chars (~8k tokens)
    # This prevents hitting the 'Tokens Per Minute' limit immediately.
    truncated_context = context_data[:30000]
    
    for attempt in range(5): 
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=f"Context: {truncated_context}\n\nQuestion: {prompt}"
            )
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                # API usually tells us how long to wait; we default to a smart delay
                wait_time = (attempt + 1) * 12  # 12s, 24s, 36s...
                st.warning(f"Quota reached (429). Cooling down for {wait_time}s... (Attempt {attempt+1}/5)")
                time.sleep(wait_time)
                continue
            raise e
    st.error("Maximum retries reached. Please wait 60 seconds for the API window to reset.")
    return None

# --- 4. SESSION MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. UI LAYOUT ---
with st.sidebar:
    st.title("Admin")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Upload Knowledge Base", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display historical messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# User Input Logic
if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # 1. Compile all document context
        knowledge_context = load_knowledge_base(uploads)
        
        # 2. Call the safe_generate retry function
        try:
            response_stream = safe_generate(user_prompt, knowledge_context)
            
            if response_stream:
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Execution Error: {e}")
