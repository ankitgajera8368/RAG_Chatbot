import os
import streamlit as st
import requests
import time
import dotenv
from config import WEB_SERVER_URL

dotenv.load_dotenv()


def streaming_inference(user_query: str, chat_history: list[str]):
    """
    Generator function that streams partial LLM responses from an API
    that returns data in an SSE-like format (one JSON chunk per line).
    Yields partial text chunks as they arrive.
    """

    url = f"{WEB_SERVER_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "query": user_query,
    }

    # The key is using stream=True in the requests.post call:
    response = requests.post(url, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        # Handle errors
        yield f"Error: {response.status_code}\n{response.text}"
        return

    # Iterate line by line.
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            yield chunk


def main():
    st.title("Chatbot to help with Ubuntu setup:")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 2) Create a placeholder for partial streaming
            partial_response_placeholder = st.empty()
            partial_response_placeholder.markdown("")
            response = ""

            for chunk in streaming_inference(prompt, st.session_state.messages):
                response += chunk
                partial_response_placeholder.markdown(response)

                # A small delay can help visualize the streaming
                time.sleep(0.1)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
