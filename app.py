import streamlit as st
import os
from google import genai
from pypdf import PdfReader
import docx
import uuid

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="Assistant AI", layout="wide", page_icon="🤖")

# Use a more robust secret check
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
elif os.getenv("GOOGLE_API_KEY"):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    st.error("🔑 API Key not found. Please add GOOGLE_API_KEY to .env or Streamlit Secrets.")
    st.stop()

# --- 2. FILE EXTRACTION UTILITIES ---

@st.cache_data(show_spinner="Reading documents...")
def extract_text_from_file(file_obj, is_path=False):
    try:
        if is_path:
            name = file_obj
            ext = os.path.splitext(name)[-1].lower()
        else:
            name = file_obj.name
            ext = os.path.splitext(name)[-1].lower()

        text = ""
        if ext == ".pdf":
            reader = PdfReader(file_obj)
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content + "\n"
        elif ext == ".docx":
            doc = docx.Document(file_obj)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            if is_path:
                with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = file_obj.getvalue().decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        return f"Error reading {name}: {str(e)}"

def get_backend_context(folder_name="documents"):
    context = ""
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            if os.path.isfile(file_path):
                context += extract_text_from_file(file_path, is_path=True) + "\n"
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
    st.rerun()

# --- 4. SIDEBAR & UI ---
with st.sidebar:
    st.title("Settings")
    if st.button("➕ Start New Chat", use_container_width=True):
        start_new_chat()
    
    st.subheader("Chat History")
    chat_ids = list(st.session_state.all_chats.keys())
    
    # Use a callback to update current_chat_id
    def on_chat_change():
        st.session_state.current_chat_id = st.session_state.selected_chat_ui

    st.selectbox(
        "Select Chat:",
        options=chat_ids,
        format_func=lambda x: st.session_state.all_chats[x]["name"],
        key="selected_chat_ui",
        on_change=on_chat_change,
        index=chat_ids.index(st.session_state.current_chat_id)
    )
    
    st.divider()
    uploaded_files = st.file_uploader("Upload Additional Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- 5. MAIN CHAT INTERFACE ---
st.title("🤖 Assistant Research Chatbot")

active_chat = st.session_state.all_chats[st.session_state.current_chat_id]

# Display historical messages
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about your documents..."):
    # Set chat name based on first question
    if not active_chat["messages"]:
        active_chat["name"] = (prompt[:25] + "...") if len(prompt) > 25 else prompt

    # Save and display user message
    active_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        # 1. Gather Context
        backend_text = get_backend_context("documents")
        uploaded_text = ""
        if uploaded_files:
            for f in uploaded_files:
                uploaded_text += extract_text_from_file(f)
        
        # Limit context to keep prompt efficient (Gemini 2.0 can take more, but 30k is a good start)
        total_context = (backend_text + uploaded_text)[:30000]
        
        # 2. Build Prompt
        full_content = (
            f"You are a helpful research assistant. Use the following context to answer the user.\n"
            f"CONTEXT:\n{total_context}\n\n"
            f"USER QUESTION: {prompt}\n\n"
            f"INSTRUCTIONS: If the answer isn't in the context, use your general knowledge but mention it. "
            f"Keep the formatting clean with markdown."
        )

        # 3. Stream the response
        try:
            placeholder = st.empty()
            full_response = ""
            
            # Using the new Google GenAI SDK syntax
            response_stream = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=full_content
            )
            
            for chunk in response_stream:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            active_chat["messages"].append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Assistant Error: {str(e)}")
