import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import docx

# --- 1. INITIAL CONFIGURATION ---
# In Streamlit Cloud, add your API key to "Secrets"
# Or replace with your key: genai.configure(api_key="YOUR_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.warning("Please add your GOOGLE_API_KEY to Streamlit Secrets to get live answers.")

# --- 2. FILE EXTRACTION UTILITY ---
def extract_text(uploaded_files):
    context = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[-1].lower()
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text: context += text
        elif ext == ".docx":
            doc = docx.Document(uploaded_file)
            context += "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            context += uploaded_file.getvalue().decode("utf-8")
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
    if st.button("➕ Start New Chat"):
        reset_chat()
    st.divider()
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 4. CHAT LOGIC ---
# Display historical messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.download_button("📥 Download Answer", msg["content"], file_name=f"answer_{i}.txt", key=f"dl_{i}")

# Handle New Input
if prompt := st.chat_input("Ask a specific question about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Get text from documents
        doc_text = extract_text(uploaded_files) if uploaded_files else "No documents uploaded."
        
        # 2. Build the AI Prompt
        # This framing ensures the Assistant gives direct, accurate answers.
        full_prompt = f"""
        You are a highly accurate Research Assistant. 
        CONTEXT FROM UPLOADED DOCUMENTS:
        {doc_text[:15000]} 
        
        USER QUESTION:
        {prompt}
        
        INSTRUCTIONS:
        - Provide a direct, accurate answer based ONLY on the context above.
        - If the answer isn't in the documents, say you don't have that specific data.
        - Do not provide code unless specifically asked for code.
        - Use professional, clinical language.
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            answer = response.text
        except Exception as e:
            answer = f"Assistant: I encountered an error processing that. Error: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Immediate Download Option
        st.download_button("📥 Download Answer", answer, file_name="latest_answer.txt", key="dl_new")
