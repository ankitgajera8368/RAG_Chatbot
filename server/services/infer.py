from typing import Dict, Generator, List, Optional
import requests
from config import LLM_MODEL_NAME, KRUTRIM_CLOUD_API_KEY
import json


def get_llm_response(
    user_query: str,
    context_chunks: list[str],
    chat_history: List[Dict[str, str]],
    stats: Optional[Dict[str, any]],
) -> Generator[str, None, None]:
    """
    Sends a request to KrutrimCloud LLM with the context and user's question.
    Returns the LLM's response string.
    """

    # Combine context chunks into a single string (if you prefer, you can store them separately)
    # context_text = "\n\n".join(context_chunks)
    context_text = ""
    for chunk in context_chunks:
        filepath = f"Filepath: {chunk['filepath']}"
        title = f'Title: {chunk["metadata"]["title"] if "title" in chunk["metadata"] else ""}'
        content = f'Context: {chunk["content"]}'

        context_text += filepath + "\n" + title + "\n" + content + "\n\n"

    # System prompt that references the context
    system_prompt = f"""
You are a knowledgeable assistant. You have been given some context information below.

Context:
{context_text}

Instructions:

1)Write your output in below format:
<your response>


**Reference**: <filepath>

Your goal is to give the best possible answer to the user based on the context and be elaborative about it.

"""

    # Build the final messages array: system prompt + existing chat history + new user query
    messages = []

    # 1) system
    messages.append({"role": "system", "content": system_prompt})

    # 2) existing chat history
    #    e.g. user -> assistant -> user -> assistant -> ...
    total_convo = len(chat_history)
    if total_convo > 5:
        max_chat = 5
    else:
        max_chat = total_convo
    for msg in chat_history[total_convo - max_chat :]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 3) the new user query
    messages.append({"role": "user", "content": user_query})

    url = "https://cloud.olakrutrim.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KRUTRIM_CLOUD_API_KEY}",
    }

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": True,  # request streaming
        "temperature": 0.75,
    }

    try:
        # The key is using stream=True in the requests.post call:
        response = requests.post(url, headers=headers, json=payload, stream=True)

        if response.status_code != 200:
            # Handle errors
            yield f"CustomInternalError: {response.status_code}\n{response.text}"
            return

        # Iterate line by line. Each line should be something like "data: { ... }"
        # until we get "[DONE]" or the stream ends
        if not stats:
            stats = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}
        full_response = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                # Typically lines start with "data: " prefix
                if chunk.startswith("data: "):
                    data_str = chunk[len("data: ") :].strip()
                    # End-of-stream sentinel
                    if data_str == "[DONE]":
                        messages.append({"role": "assistant", "content": full_response})
                        for msg in messages:
                            if msg["role"] != "system":
                                chat_history.append(
                                    {"role": msg["role"], "content": msg["content"]}
                                )
                        break

                    # Parse JSON to extract partial content
                    try:
                        data_obj = json.loads(data_str)
                        # Per typical SSE from LLMs: data_obj["choices"][0]["delta"]["content"]
                        if data_obj["choices"]:
                            content = data_obj["choices"][0]["delta"].get("content", "")
                            full_response += content
                        if "usage" in data_obj and data_obj["usage"]:
                            stats["completion_tokens"] = data_obj["usage"][
                                "completion_tokens"
                            ]
                            stats["prompt_tokens"] = data_obj["usage"]["prompt_tokens"]
                            stats["total_tokens"] = data_obj["usage"]["total_tokens"]
                        # TODO: add pricing for each call for logging purpose
                        yield content
                    except json.JSONDecodeError:
                        # If it's not valid JSON, just yield raw chunk
                        yield chunk
    except Exception as exc:
        return f"Error calling LLM: {exc}"
