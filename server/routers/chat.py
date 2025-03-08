import json
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse

from services.indexing import build_or_load_index
from services.retrieval import retrieve_top_k
from services.infer import get_llm_response
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DATASET_PATH,
    MAX_CHARS,
    CHAT_HISTORY,
)

router = APIRouter(prefix="/chat", tags=["Chat"])

# For demonstration, we’ll do the index building/loading once at start-up
# But you can rebuild or reload as needed, or store it as a global variable.
index, chunk_texts = build_or_load_index(
    DATASET_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    max_chars=MAX_CHARS,
)  # if you already have an index on disk


@router.post("/completions")
def chat_with_docs(
    query: str = Body(..., description="User query"),
    top_k: int = Body(3, description="Number of relevant chunks to retrieve"),
):
    """
    Retrieves the top_k relevant chunks from the index and queries the LLM.
    Returns the LLM response.
    """

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Retrieve relevant chunks
    retrieved_chunks = retrieve_top_k(query, index, chunk_texts, top_k=top_k)
    print(f"Retrieved Chunks: \n{json.dumps(retrieved_chunks, indent=4)}")

    def response_generator():
        try:
            response = ""
            for response_chunk in get_llm_response(
                query, retrieved_chunks, CHAT_HISTORY, stats={}
            ):
                if response_chunk.startswith("CustomInternalError:"):
                    raise HTTPException(
                        status_code=500,
                        detail="Could not receive response from LLM service!",
                    )
                response += response_chunk
                yield response_chunk
                # yield json.dumps({"content": response_chunk}, ensure_ascii=False) + "\n"
            print(f"Response: {response}")
        except Exception as exc:
            # Handle error gracefully:
            # - Log it,
            # - Possibly yield an error message to the client,
            # - Or terminate the stream.
            # Example: yield an error text
            error_msg = f"\n[Error encountered]: {str(exc)}\n"
            yield error_msg

    return StreamingResponse(response_generator(), media_type="text/plain")
