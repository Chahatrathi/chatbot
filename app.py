import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Instant AI", layout="wide")

# Initialize Backend once and cache it for speed
@st.cache_resource
def get_backend():
    return ChatBackend()

backend = get_backend()

st.title("⚡ Instant Memory Chatbot")

# Sidebar for immediate access to data
with st.sidebar:
    st.header("Data Archive")
    chat_log = backend.db.get_chat_as_text()
    st.download_button("📥 Download All Chats", chat_log, "chat_history.txt", use_container_width=True)
    
    if st.button("🗑️ Clear Database", use_container_width=True):
        backend.db.conn.execute("DELETE FROM messages")
        backend.db.conn.commit()
        st.rerun()

# Display History immediately on load
for role, content in backend.db.get_all_history():
    with st.chat_message(role):
        st.markdown(content)

# Instant Input
if prompt := st.chat_input("How can I help you right now?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_generator = backend.get_streaming_response(prompt)
        full_response = st.write_stream(response_generator)
        
    # Persistent saving in the background
    backend.db.save_message("user", prompt)
    backend.db.save_message("assistant", full_response)
    st.rerun()
