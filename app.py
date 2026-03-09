import streamlit as st
import os
from google import genai
from pypdf import PdfReader
import docx
import uuid

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
# Initialize the multi-chat storage
if "all_chats" not in st.session_state:
    # Format: { "uuid": {"name": "Chat Name", "messages": []} }
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
    
    # New Chat Button
    if st.button("➕ Start New Chat"):
        start_new_chat()
    
    st.divider()
    
    # Select from History
    chat_options = list(st.session_state.all_chats.keys())
    selected_chat = st.selectbox(
        "Select an old chat:",
        options=chat_options,
        format_func=lambda x: st.session_state.all_chats[x]["name"],
        index=chat_options.index(st.session_state.current_chat_id)
    )
    st.session_state.current_chat_id = selected_chat

    st.divider()
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 5. CHAT LOGIC ---
current_chat = st.session_state.all_chats[st.session_state.current_chat_id]

# Display history of the selected chat
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    # If it's the first message, rename the chat based on the prompt
    if not current_chat["messages"]:
        current_chat["name"] = prompt[:20] + "..."

    # Add user message
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        doc_text = extract_text(uploaded_files) if uploaded_files else "No documents uploaded."
        full_content = f"Context from documents: {doc_text[:25000]}\n\nUser Question: {prompt}\n\nAnswer directly."

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=full_content
            )
            answer = response.text
            st.markdown(answer)
            current_chat["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Assistant Error: {e}")
