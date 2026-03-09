import streamlit as st
import os
from google import genai
from pypdf import PdfReader
import docx
import uuid
import time
from google.genai.errors import ClientError

# --- 1. INITIAL CONFIGURATION ---
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(
        api_key=st.secrets["GOOGLE_API_KEY"],
        http_options={'api_version': 'v1'}
    )
else:
    st.error("Please add GOOGLE_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- 2. SESSION MANAGEMENT ---
if "all_chats" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.all_chats = {
        initial_id: {"name": "New Chat", "messages": []}
    }
    st.session_state.current_chat_id = initial_id

def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

# --- 3. FILE EXTRACTION ---
def extract_text(uploaded_files):
    context = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[-1].lower()
        try:
            if ext == ".pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: context += text + "\n"
            elif ext == ".docx":
                doc = docx.Document(uploaded_file)
                context += "\n".join([p.text for p in doc.paragraphs]) + "\n"
            elif ext == ".txt":
                context += uploaded_file.getvalue().decode("utf-8") + "\n"
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
    return context

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Assistant Research Bot", layout="wide")
st.title("🤖 Assistant Research Chatbot")

with st.sidebar:
    st.header("Chat History")
    if st.button("➕ Start New Chat"):
        start_new_chat()
    
    chat_ids = list(st.session_state.all_chats.keys())
    selected_chat = st.selectbox(
        "History:", 
        options=chat_ids, 
        format_func=lambda x: st.session_state.all_chats[x]["name"],
        index=chat_ids.index(st.session_state.current_chat_id)
    )
    st.session_state.current_chat_id = selected_chat
    st.divider()
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 5. CHAT LOGIC ---
active_chat = st.session_state.all_chats[st.session_state.current_chat_id]

for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    if not active_chat["messages"]:
        active_chat["name"] = prompt[:30] + "..."

    active_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        doc_text = extract_text(uploaded_files) if uploaded_files else ""
        full_prompt = f"Context: {doc_text[:20000]}\n\nQuestion: {prompt}\n\nAnswer concisely. Use general knowledge if context is empty."

        # Implementation of Exponential Backoff for 429 Errors
        max_retries = 3
        retry_delay = 5 # seconds
        success = False

        for i in range(max_retries):
            try:
                # Using 2.5-flash-lite for higher free-tier limits
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite", 
                    contents=full_content
                )
                answer = response.text
                st.markdown(answer)
                active_chat["messages"].append({"role": "assistant", "content": answer})
                success = True
                break
            except Exception as e:
                if "429" in str(e) and i < max_retries - 1:
                    st.warning(f"Quota busy. Retrying in {retry_delay}s... (Attempt {i+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Double the wait time
                else:
                    st.error(f"Assistant Error: {e}")
                    break
