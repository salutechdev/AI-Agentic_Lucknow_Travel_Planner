import streamlit as st
import requests
import os
import uuid

# --- CONFIGURATION ---
st.set_page_config(page_title="Advanced AI Lucknow Tour Guide", page_icon="🗺️", layout="wide")

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

st.title("🗺️ Advanced AI Lucknow Tour Guide")
st.caption("⚡ Powered by a FastAPI backend, Groq, and RAG")

# Ensure a unique, persistent session tracking ID exists for this instance
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display current session history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user interaction input
if prompt := st.chat_input("Plan my trip to Lucknow..."):
    # Add user message to local history and render it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dispatch request to FastAPI server
    with st.chat_message("assistant"):
        with st.spinner("The AI planner is thinking..."):
            try:
                # Build payload matching the updated Pydantic model contract
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }

                response = requests.post(BACKEND_URL, json=payload)
                response.raise_for_status()

                backend_response_data = response.json()
                backend_response = backend_response_data.get("response", "No response text received from backend.")

                st.markdown(backend_response)
                st.session_state.chat_history.append({"role": "assistant", "content": backend_response})

            except requests.exceptions.RequestException as e:
                error_message = (f"Could not connect to or get a valid response from the backend. "
                                 f"Please ensure the FastAPI server is running and accessible at {BACKEND_URL}. "
                                 f"Error: {e}")
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})


