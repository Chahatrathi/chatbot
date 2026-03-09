import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Secure Internal Bot", layout="wide")

@st.cache_resource
def get_backend():
    return ChatBackend()

backend = get_backend()

# Get the unique ID for this browser tab session
from streamlit.runtime.scriptrunner import get_script_run_ctx
ctx = get_script_run_ctx()
session_id = ctx.session_id if ctx else "default_session"

st.title("🛡️ Private Internal Knowledge Bot")
st.caption(f"Connected as Session: {session_id[:8]}...")

# Sidebar - Specific to this user
with st.sidebar:
    if st.button("🗑️ Clear My History"):
        backend.db.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        backend.db.conn.commit()
        st.success("Your private history cleared!")
        st.rerun()

# Display Private Messages
for role, content in backend.db.get_session_history(session_id):
    with st.chat_message(role):
        st.markdown(content)

# Handle Chat
if prompt := st.chat_input("Ask a follow-up question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Pass session_id to ensure the bot remembers ONLY this user
            response_generator = backend.get_streaming_response(prompt, session_id)
            full_response = st.write_stream(response_generator)
            
            backend.db.save_message(session_id, "user", prompt)
            backend.db.save_message(session_id, "assistant", full_response)
            st.rerun()
        except Exception as e:
            st.error(f"System Error: {e}")
