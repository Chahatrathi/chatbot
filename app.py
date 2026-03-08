import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Persistent Gemini", page_icon="💾")

st.title("💾 Persistent Memory Chatbot")
st.caption("All data is stored in the backend database.")

api_key = st.sidebar.text_input("Google API Key", type="password")

if api_key:
    # Initialize Backend
    if "backend" not in st.session_state:
        st.session_state.backend = ChatBackend(api_key)
    
    # Load and display persistent history from DB
    history = st.session_state.backend.db.get_all_history()
    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)

    # Chat Input
    if prompt := st.chat_input("Say something..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = st.session_state.backend.get_response(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.info("Please enter your API key to load your chat history.")
