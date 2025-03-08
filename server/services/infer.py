from typing import Dict, Generator, List, Optional
import requests
from config.config import (
    LLM_MODEL_NAME,
    KRUTRIM_CLOUD_API_KEY,
    LLM_PRICE_PER_MILLION_TOKEN,
    MAX_CONVERSATION_MSGS,
)
import json
import logging
import config.logging_config

logger = logging.getLogger(__name__)


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

    # Combine context chunks into a single string along with filepath and title of each chunk
    context_text = ""
    for chunk in context_chunks:
        filepath = f"Filepath: {chunk['filepath']}"
        title = f'Title: {chunk["metadata"]["title"] if "title" in chunk["metadata"] else ""}'
        content = f'Context: {chunk["content"]}'

        context_text += filepath + "\n" + title + "\n" + content + "\n\n"

    # System prompt that references the context retrieved from the vector store
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

    # 1) system prompt
    messages.append({"role": "system", "content": system_prompt})

    # 2) existing chat history
    #    e.g. user -> assistant -> user -> assistant -> ...
    total_convo = len(chat_history)
    # TODO: Chat history truncation logic can be improved by tracking input tokens and LLM's token length
    if total_convo > MAX_CONVERSATION_MSGS:
        logger.info(
            f"Truncating the conversation history, keeping last {MAX_CONVERSATION_MSGS} messages"
        )
        max_chat = MAX_CONVERSATION_MSGS
    else:
        max_chat = total_convo
    for msg in chat_history[total_convo - max_chat :]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 3) the new user query
    messages.append({"role": "user", "content": user_query})

    logger.info(f"Query: {user_query}")

    url = "https://cloud.olakrutrim.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KRUTRIM_CLOUD_API_KEY}",
    }

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.75,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)

        if response.status_code != 200:
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
                        if data_obj["choices"]:
                            content = data_obj["choices"][0]["delta"].get("content", "")
                            full_response += content
                        if "usage" in data_obj and data_obj["usage"]:
                            stats["completion_tokens"] = data_obj["usage"][
                                "completion_tokens"
                            ]
                            stats["prompt_tokens"] = data_obj["usage"]["prompt_tokens"]
                            stats["total_tokens"] = data_obj["usage"]["total_tokens"]
                            logger.info(
                                f"Input tokens : {stats["prompt_tokens"]}\nOutput tokens : {stats["completion_tokens"]}\nCost incurred for current query: {round((int(stats["total_tokens"])*LLM_PRICE_PER_MILLION_TOKEN)/1000000, 3)} INR"
                            )
                        yield content
                    except json.JSONDecodeError:
                        # If it's not valid JSON, just yield raw chunk
                        logger.debug("If it's not valid JSON, just yielded raw chunk")
                        yield chunk
    except Exception as exc:
        error_msg = f"Error calling LLM: {exc}"
        logger.error(error_msg)
        return error_msg
