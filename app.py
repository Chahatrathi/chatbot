import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="General AI Chatbot", page_icon="🤖")
st.title("🤖 General AI Assistant")

# Initialize Backend with Secrets
if "backend" not in st.session_state:
    st.session_state.backend = ChatBackend(st.secrets["GOOGLE_API_KEY"])

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls and download
with st.sidebar:
    st.header("Chat Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        st.write("---")
        csv_data = st.session_state.backend.export_history(st.session_state.messages, format="csv")
        st.download_button(
            label="📥 Download Chat History (CSV)",
            data=csv_data,
            file_name="chat_history.csv",
            mime="text/csv"
        )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    
    # Generate response via Backend
    with st.spinner("Thinking..."):
        try:
            full_response = st.session_state.backend.generate_response(
                prompt, st.session_state.messages
            )
            
            # Show assistant response
            with st.chat_message("assistant"):
                st.markdown(full_response)
            
            # Save to Session History
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")
