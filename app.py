import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="AI Vault", layout="wide")

# Initialize State
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "setup_finished" not in st.session_state:
    st.session_state.setup_finished = False

st.title("💬 Secure Memory Chatbot")

# Sidebar for the Download Button
if st.session_state.setup_finished:
    with st.sidebar:
        st.header("Archive")
        chat_data = st.session_state.backend.db.get_chat_as_text()
        st.download_button("📥 Download Conversations", chat_data, "history.txt")

# --- CHAT LOGIC ---

# Step 1: Request API Key via Chat
if not st.session_state.setup_finished:
    with st.chat_message("assistant"):
        st.markdown("Hello! To get started, please paste your **Google API Key** below. (I won't save it outside this session).")
    
    if key_input := st.chat_input("Paste your API key here..."):
        # Simple validation: Gemini keys usually start with 'AIza'
        if key_input.startswith("AIza") and len(key_input) > 20:
            st.session_state.api_key = key_input
            st.session_state.backend = ChatBackend(key_input)
            st.session_state.setup_finished = True
            st.success("API Key accepted!")
            st.rerun()
        else:
            st.error("That doesn't look like a valid Google API key. Please try again.")

# Step 2: Normal Chat Mode
else:
    # Display historical chats from the DB
    for role, content in st.session_state.backend.db.get_all_history():
        with st.chat_message(role):
            st.markdown(content)

    # Process new input
    if prompt := st.chat_input("How can I help you today?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.backend.get_response(prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}. Your API key might be invalid or expired.")
