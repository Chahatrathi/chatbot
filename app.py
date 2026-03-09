import streamlit as st
import os
import time
import uuid
import docx
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="Pro Research Assistant", layout="wide", page_icon="🤖")

def get_client():
    # Priority: Streamlit Secrets > Environment Variables
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 API Key missing! Add it to Streamlit Secrets or your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

# --- 2. DATA EXTRACTION UTILITIES ---
@st.cache_data(show_spinner=False)
def extract_text(file_source, is_path=False):
    """Robustly extracts text from PDF, DOCX, and TXT files."""
    try:
        name = file_source if is_path else file_source.name
        ext = os.path.splitext(name)[-1].lower()
        text = ""
        if ext == ".pdf":
            reader = PdfReader(file_source)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif ext == ".docx":
            doc = docx.Document(file_source)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = file_source.getvalue().decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        return f"Error reading {name}: {e}"

def load_combined_context(uploads):
    """Combines text from the 'documents/' folder and user uploads."""
    context = ""
    # Auto-load from project 'documents' folder
    if os.path.exists("documents"):
        for filename in os.listdir("documents"):
            path = os.path.join("documents", filename)
            if os.path.isfile(path):
                context += extract_text(path, is_path=True) + "\n"
    # Load from the sidebar uploader
    if uploads:
        for f in uploads:
            context += extract_text(f) + "\n"
    return context

# --- 3. THE ERROR ERADICATOR (Retry Logic) ---
def safe_generate(prompt, context_data):
    """Eradicates 429 errors by catching them and retrying with exponential backoff."""
    # TPM Management: Truncate context to ~30k chars (~8k tokens) to stay under free limits
    # This prevents hitting the 'Tokens Per Minute' cap immediately.
    truncated_context = context_data[:30000]
    
    # We try 5 times with increasing wait times
    for attempt in range(5): 
        try:
            return client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=f"Context: {truncated_context}\n\nQuestion: {prompt}"
            )
        except Exception as e:
            # Look for the 429 RESOURCE_EXHAUSTED error code in the exception
            if "429" in str(e):
                # Wait 8s, 16s, 24s... giving the API window time to reset
                wait_time = (attempt + 1) * 8  
                st.warning(f"Quota reached. Auto-retrying in {wait_time}s... (Attempt {attempt+1}/5)")
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
    st.title("Admin Panel")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.subheader("Knowledge Base")
    uploads = st.file_uploader("Add research papers or data:", type=["pdf", "txt", "docx"], accept_multiple_files=True)

st.title("🤖 Assistant Research Chatbot")

# Display historical messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input Logic
if user_prompt := st.chat_input("Ask a question about your research..."):
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # 1. Compile all document context
        full_context = load_combined_context(uploads)
        
        # 2. Call the safe_generate retry function
        try:
            response_stream = safe_generate(user_prompt, full_context)
            
            if response_stream:
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                
                # Finalize the response
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.info("💡 Pro Tip: If you see 'Free Tier' errors often, ensure your project in AI Studio has billing enabled.")
