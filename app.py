import streamlit as st
import os
from google import genai
from pypdf import PdfReader
import docx

# --- 1. INITIAL CONFIGURATION ---
# Initialize the client using the stable v1 API version
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(
        api_key=st.secrets["GOOGLE_API_KEY"],
        http_options={'api_version': 'v1'} # Force stable API version
    )
else:
    st.error("Please add GOOGLE_API_KEY to your Streamlit Secrets.")
    st.stop()

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
            st.error(f"Error reading {uploaded_file.name}: {e}")
    return context

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_chat():
    st.session_state.messages = []
    st.rerun()

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Assistant AI", layout="wide")
st.title("🤖 Assistant Research Chatbot")

with st.sidebar:
    st.header("Settings")
    if st.button("➕ Start New Chat"):
        reset_chat()
    st.divider()
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 5. CHAT LOGIC ---
# Display messages directly (No download buttons)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about equity or your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        doc_text = extract_text(uploaded_files) if uploaded_files else "No documents uploaded."
        
        # Build prompt: Context window is large enough for text datasets
        full_content = f"Context from documents: {doc_text[:25000]}\n\nUser Question: {prompt}\n\nAnswer directly and concisely."

        try:
            # FIX: Updated to supported stable model 'gemini-2.5-flash'
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=full_content
            )
            answer = response.text
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Assistant Error: {e}")
