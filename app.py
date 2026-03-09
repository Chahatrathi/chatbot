import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import docx

# --- 1. INITIAL CONFIGURATION ---
# Replace 'YOUR_API_KEY' with your actual key or use st.secrets
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    # Manual fallback for local testing
    genai.configure(api_key="YOUR_API_KEY_HERE")

# --- 2. FILE EXTRACTION UTILITY ---
def extract_text(uploaded_files):
    context = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[-1].lower()
        try:
            if ext == ".pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    content = page.extract_text()
                    if content: context += content + "\n"
            elif ext == ".docx":
                doc = docx.Document(uploaded_file)
                context += "\n".join([para.text for para in doc.paragraphs]) + "\n"
            elif ext == ".txt":
                context += uploaded_file.getvalue().decode("utf-8", errors="ignore") + "\n"
        except Exception as e:
            st.error(f"Could not read {uploaded_file.name}: {e}")
    return context

# --- 3. SESSION STATE & UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_chat():
    st.session_state.messages = []
    st.rerun()

st.set_page_config(page_title="Assistant AI", layout="wide")
st.title("🤖 Assistant Research Chatbot")

with st.sidebar:
    st.header("Settings")
    if st.button("➕ Start New Chat"):
        reset_chat()
    st.divider()
    uploaded_files = st.file_uploader("Upload Project Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 4. CHAT INTERFACE ---

# Display message history (Buttons removed)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle New Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to UI and state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        doc_text = extract_text(uploaded_files) if uploaded_files else "No specific documents provided."
        
        # Refined Prompt for Directness
        full_prompt = f"""
        Context: {doc_text[:15000]}
        
        Question: {prompt}
        
        Instruction: Provide a direct, factual, and accurate response based on the context. 
        If the context doesn't contain the answer, use your general knowledge but prioritize document data.
        Be concise and do not include any meta-talk or technical code unless requested.
        """

        try:
            # Using 'models/gemini-1.5-flash' explicitly to resolve the 404 error
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            answer = response.text
            
            # Direct display
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"Assistant: I encountered an error. Please check your API key or connection. Details: {e}"
            st.error(error_msg)
