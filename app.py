import streamlit as st
import os
import time
import docx
import uuid
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key not found! Please check your Streamlit Secrets or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. THE ERROR ERADICATOR (Your safe_generate function) ---
def safe_generate(prompt, context):
    """Handles 429 errors by waiting and retrying with exponential backoff."""
    # TPM Management: Truncate context to ~8k tokens to prevent quota exhaustion
    truncated_context = context[:30000]
    
    for attempt in range(5):  # Try up to 5 times
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=f"Context: {truncated_context}\n\nQuestion: {prompt}"
            )
        except Exception as e:
            # Check for the 429 Resource Exhausted error
            if "429" in str(e):
                wait_time = (attempt + 1) * 10  # Increased wait for better stability
                st.warning(f"Quota hit. Retrying in {wait_time}s... (Attempt {attempt+1}/5)")
                time.sleep(wait_time)
                continue
            raise e
    st.error("Maximum retries reached. Please wait a minute before trying again.")
    return None

# --- 3. DATA EXTRACTION UTILITIES ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    try:
        ext = os.path.splitext(file_source if is_path else file_source.name)[-1].lower()
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
    except Exception as e:
        return f"Error reading file: {e}"
    return ""

def load_knowledge_base(uploads):
    full_text = ""
    # Load from the 'documents/' folder automatically
    if os.path.exists("documents"):
        for filename in os.listdir("documents"):
            path = os.path.join("documents", filename)
            if os.path.isfile(path):
                full_text += extract_text(path, is_path=True) + "\n"
    # Load from the manual uploader
    if uploads:
        for f in uploads:
            full_text += extract_text(f) + "\n"
    return full_text

# --- 4. SESSION MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. UI LAYOUT ---
with st.sidebar:
    st.title("Admin Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    uploads = st.file_uploader("Upload Additional Files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Logic
if user_prompt := st.chat_input("Ask a question about your data..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            full_response = ""
            
            # Step 1: Get all text data
            knowledge_context = load_knowledge_base(uploads)
            
            # Step 2: Use your safe_generate function
            response_stream = safe_generate(user_prompt, knowledge_context)
            
            if response_stream:
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                # Step 3: Save to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
