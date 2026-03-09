import streamlit as st
import os
from google import genai
from pypdf import PdfReader
import docx
import uuid
import time

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
    st.session_state.all_chats[new_id] = {"name": f"Chat {len(st.session_state.all_chats) + 1}", "messages": []}
    st.session_state.current_chat_id = new_id

# --- 3. FILE EXTRACTION UTILITY ---
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

# --- 4. UI LAYOUT & SIDEBAR ---
st.set_page_config(page_title="Assistant AI", layout="wide")
st.title("🤖 Assistant Research Chatbot")

with st.sidebar:
    st.header("Chat History")
    if st.button("➕ Start New Chat"):
        start_new_chat()
    
    st.divider()
    chat_options = list(st.session_state.all_chats.keys())
    selected_chat = st.selectbox(
        "Select a conversation:",
        options=chat_options,
        format_func=lambda x: st.session_state.all_chats[x]["name"],
        index=chat_options.index(st.session_state.current_chat_id)
    )
    st.session_state.current_chat_id = selected_chat

    st.divider()
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 5. CHAT LOGIC ---
current_chat = st.session_state.all_chats[st.session_state.current_chat_id]

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    if not current_chat["messages"]:
        current_chat["name"] = prompt[:30] + "..."

    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        doc_text = extract_text(uploaded_files) if uploaded_files else "No documents uploaded."
        full_content = f"Context: {doc_text[:15000]}\n\nQuestion: {prompt}\n\nAnswer directly."

        try:
            # Reverting to the most stable model name for v1 API
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=full_content
            )
            answer = response.text
            st.markdown(answer)
            current_chat["messages"].append({"role": "assistant", "content": answer})
            
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Rate limit reached. Retrying in 5 seconds...")
                time.sleep(5)
                try:
                    response = client.models.generate_content(model="gemini-1.5-flash", contents=full_content)
                    st.markdown(response.text)
                    current_chat["messages"].append({"role": "assistant", "content": response.text})
                except:
                    st.error("🚨 Quota exceeded. Please wait a minute before trying again.")
            else:
                st.error(f"Assistant Error: {e}")
