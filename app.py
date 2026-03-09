import streamlit as st
import os
from io import BytesIO

# --- ROBUST IMPORTS ---
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Library 'pypdf' not found. Please ensure it is in requirements.txt")

try:
    import docx
except ImportError:
    st.error("Library 'python-docx' not found. Please ensure it is in requirements.txt")

# --- 1. FILE EXTRACTION UTILITIES ---

def extract_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        elif ext == ".docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"
    return text

# --- 2. SESSION STATE SETUP ---

if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_chat():
    st.session_state.messages = []
    st.rerun()

# --- 3. UI LAYOUT ---

st.set_page_config(page_title="Assistant Research Bot", layout="wide")
st.title("🤖 Assistant Chatbot")

with st.sidebar:
    st.header("Control Panel")
    # New Chat Option
    if st.button("➕ Start New Chat"):
        reset_chat()
    
    st.divider()
    
    # File Upload Section
    uploaded_files = st.file_uploader(
        "Upload Project Documents", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files loaded successfully.")

# --- 4. CHAT INTERFACE ---

# Display message history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Individual Download for Assistant answers
        if message["role"] == "assistant":
            st.download_button(
                label="📥 Download this answer",
                data=message["content"],
                file_name=f"assistant_response_{i}.txt",
                mime="text/plain",
                key=f"dl_{i}" 
            )

# Chat Input Logic
if prompt := st.chat_input("How can the Assistant help you today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        # Compile text from all uploaded documents for context
        document_context = ""
        if uploaded_files:
            for f in uploaded_files:
                document_context += extract_text(f) + "\n"
        
        # This is where your AI Logic/API call would go. 
        # For now, it simulates a response based on the "Assistant" identity.
        response_text = f"Assistant: Based on the documents provided, here is the information regarding '{prompt}'.\n\n[Analysing context...]\n\nI have reviewed the uploaded files and found relevant data to answer your query. Please let me know if you need further details."
        
        st.markdown(response_text)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Individual Download for this specific new answer
        st.download_button(
            label="📥 Download this answer",
            data=response_text,
            file_name=f"assistant_response_latest.txt",
            mime="text/plain",
            key="dl_latest"
        )
