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
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. FILE EXTRACTION UTILITIES ---

@st.cache_data(show_spinner=False)
def extract_text_from_file(file_path_or_obj):
    try:
        if isinstance(file_path_or_obj, str):
            name = file_path_or_obj
            is_path = True
        else:
            name = file_path_or_obj.name
            is_path = False

        ext = os.path.splitext(name)[-1].lower()
        text = ""
        if ext == ".pdf":
            f = open(file_path_or_obj, "rb") if is_path else file_path_or_obj
            reader = PdfReader(f)
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content + "\n"
        elif ext == ".docx":
            f = file_path_or_obj
            doc = docx.Document(f)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_path_or_obj, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = file_path_or_obj.getvalue().decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        return ""

def get_backend_context(folder_name="documents"):
    context = ""
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            context += extract_text_from_file(file_path) + "\n"
    return context

# --- 3. SESSION MANAGEMENT ---
if "all_chats" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.all_chats = {initial_id: {"name": "New Chat", "messages": []}}
    st.session_state.current_chat_id = initial_id

def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Assistant AI", layout="wide")
st.title("🤖 Assistant Research Chatbot")

with st.sidebar:
    st.header("History")
    if st.button("➕ Start New Chat"):
        start_new_chat()
    
    chat_ids = list(st.session_state.all_chats.keys())
    selected_chat = st.selectbox("Previous Chats:", options=chat_ids, 
                                 format_func=lambda x: st.session_state.all_chats[x]["name"],
                                 index=chat_ids.index(st.session_state.current_chat_id))
    st.session_state.current_chat_id = selected_chat
    st.divider()
    uploaded_files = st.file_uploader("Upload Additional Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

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
        backend_text = get_backend_context("documents")
        uploaded_text = ""
        if uploaded_files:
            for f in uploaded_files:
                uploaded_text += extract_text_from_file(f)
        
        total_context = (backend_text + uploaded_text)[:15000]
        
        full_content = f"CONTEXT: {total_context}\n\nQUESTION: {prompt}\n\nINSTRUCTION: Answer using context. If empty, use general knowledge. Be direct."

        try:
            # Using stable gemini-2.0-flash
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=full_content
            )
            answer = response.text
            st.markdown(answer)
            active_chat["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Assistant Error: {e}")
