import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Internal Knowledge Bot", layout="wide")

@st.cache_resource
def get_backend():
    return ChatBackend()

# Initialize the backend
try:
    backend = get_backend()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

st.title("🛡️ Internal Knowledge Chatbot")

# Sidebar Logic
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        backend.db.conn.execute("DELETE FROM messages")
        backend.db.conn.commit()
        st.success("History Cleared!")
        st.rerun()
    
    # Generate download data
    history = backend.db.get_all_history()
    chat_log = "\n".join([f"{r.upper()}: {c}" for r, c in history])
    st.download_button("📥 Download Chat Log", chat_log, "chat_log.txt", use_container_width=True)

# Display Messages from DB
for role, content in backend.db.get_all_history():
    with st.chat_message(role):
        st.markdown(content)

# Handle Input
if prompt := st.chat_input("Ask about the documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_generator = backend.get_streaming_response(prompt)
            full_response = st.write_stream(response_generator)
            
            # Save interaction to DB
            backend.db.save_message("user", prompt)
            backend.db.save_message("assistant", full_response)
            
            # Refresh to show new message properly
            st.rerun()
        except Exception as e:
            st.error(f"Assistant Error: {e}")
