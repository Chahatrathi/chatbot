import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Chat Archive", page_icon="💾")

st.title("💾 Chatbot with Persistent Memory")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    # Download Logic
    if "backend" in st.session_state:
        st.subheader("Archive")
        chat_text = st.session_state.backend.db.get_chat_as_text()
        
        st.download_button(
            label="📥 Download Chats (.txt)",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )
        
        if st.button("🗑️ Clear History"):
            # Optional: Add logic here to clear the DB if you want
            st.warning("Feature not yet implemented: Requires SQL Delete")

if api_key:
    if "backend" not in st.session_state:
        st.session_state.backend = ChatBackend(api_key)
    
    # Load and display persistent history from backend
    history = st.session_state.backend.db.get_all_history()
    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Processing..."):
            response = st.session_state.backend.get_response(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Rerurn to update the download button data immediately
        st.rerun()
else:
    st.info("Enter your API key in the sidebar to load your history and start chatting.")
