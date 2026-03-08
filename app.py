import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Internal Knowledge Bot", layout="wide")

@st.cache_resource
def get_backend():
    return ChatBackend()

backend = get_backend()

st.title("🛡️ Internal Knowledge Chatbot")

# Sidebar
with st.sidebar:
    if st.button("🗑️ Clear History"):
        backend.db.conn.execute("DELETE FROM messages")
        backend.db.conn.commit()
        st.rerun()
    
    chat_log = "\n".join([f"{r.upper()}: {c}" for r, c in backend.db.get_all_history()])
    st.download_button("📥 Download Log", chat_log, "history.txt")

# Main Chat
for role, content in backend.db.get_all_history():
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask about your documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_generator = backend.get_streaming_response(prompt)
            full_response = st.write_stream(response_generator)
            backend.db.save_message("user", prompt)
            backend.db.save_message("assistant", full_response)
            st.rerun()
        except Exception as e:
            st.error(f"Safety/API Error: This topic might be restricted by the provider. Details: {e}")
