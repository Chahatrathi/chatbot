import streamlit as st
import os
from pypdf import PdfReader
import docx
from io import BytesIO

# --- 1. FILE EXTRACTION UTILITIES ---

def extract_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    text = ""
    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif ext == ".docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        text = uploaded_file.getvalue().decode("utf-8")
    return text

# --- 2. SESSION STATE SETUP ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to start a new chat
def reset_chat():
    st.session_state.messages = []
    st.rerun()

# --- 3. UI LAYOUT ---

st.title("Assistant Chatbot")

with st.sidebar:
    st.header("Tools")
    # New Chat Option
    if st.button("➕ New Chat"):
        reset_chat()
    
    st.divider()
    
    # File Upload Section
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, Word, Text)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded!")

# --- 4. CHAT INTERFACE ---

# Display message history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add a download button ONLY for assistant answers
        if message["role"] == "assistant":
            st.download_button(
                label="📥 Download this answer",
                data=message["content"],
                file_name=f"answer_{i}.txt",
                mime="text/plain",
                key=f"dl_{i}" # Unique key for every button
            )

# Chat Input Logic
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        context = ""
        if uploaded_files:
            for f in uploaded_files:
                context += extract_text(f) + "\n"
        
        # Placeholder for AI logic - Replace with your model call
        response = f"Assistant: I have analyzed your files. You asked: '{prompt}'. Here is the data-driven answer based on your documents."
        
        st.markdown(response)
        
        # Add message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Individual Download for the new answer
        st.download_button(
            label="📥 Download this answer",
            data=response,
            file_name=f"answer_{len(st.session_state.messages)}.txt",
            mime="text/plain",
            key=f"dl_new"
        )
