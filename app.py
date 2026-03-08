import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Internal Knowledge Bot", layout="wide")

# Cache the backend to ensure documents are only read once (improves speed)
@st.cache_resource
def get_backend():
    return ChatBackend()

backend = get_backend()

st.title("🛡️ Internal Knowledge Chatbot")
st.info("This bot is powered by internal backend documents.")

# Sidebar for history management
with st.sidebar:
    if st.button("🗑️ Reset Chat"):
        backend.db.conn.execute("DELETE FROM messages")
        backend.db.conn.commit()
        st.rerun()
    
    chat_log = "\n".join([f"{r}: {c}" for r, c in backend.db.get_all_history()])
    st.download_button("📥 Export Logs", chat_log, "logs.txt")

# Display Conversation
for role, content in backend.db.get_all_history():
    with st.chat_message(role):
        st.markdown(content)

# Interaction
if prompt := st.chat_input("Ask a question about the internal documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_generator = backend.get_streaming_response(prompt)
        full_response = st.write_stream(response_generator)
        
    backend.db.save_message("user", prompt)
    backend.db.save_message("assistant", full_response)
    st.rerun()