import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Fast Gemini", layout="wide")

# Persistent storage for the backend instance
if "backend" not in st.session_state:
    st.session_state.backend = None
if "setup_finished" not in st.session_state:
    st.session_state.setup_finished = False

st.title("⚡ High-Speed Gemini Chat")

# Sidebar
if st.session_state.setup_finished:
    with st.sidebar:
        st.header("Options")
        chat_log = st.session_state.backend.db.get_chat_as_text()
        st.download_button("📥 Export History", chat_log, "chat.txt", use_container_width=True)
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.setup_finished = False
            st.rerun()

# --- CHAT INTERFACE ---

if not st.session_state.setup_finished:
    with st.chat_message("assistant"):
        st.write("Please provide your API key to begin.")
    
    if key_input := st.chat_input("Enter Key..."):
        if key_input.startswith("AIza"):
            st.session_state.backend = ChatBackend(key_input)
            st.session_state.setup_finished = True
            st.rerun()
        else:
            st.error("Invalid Key Format.")

else:
    # Display History
    for role, content in st.session_state.backend.db.get_all_history():
        with st.chat_message(role):
            st.markdown(content)

    # Handle Input with Streaming
    if prompt := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Stream the response chunk by chunk
            response_generator = st.session_state.backend.get_streaming_response(prompt)
            full_response = st.write_stream(response_generator)
            
        # Save to DB only AFTER the stream finishes
        st.session_state.backend.db.save_message("user", prompt)
        st.session_state.backend.db.save_message("assistant", full_response)
        
        # Minimal rerun to update sidebar download data
        st.rerun()
