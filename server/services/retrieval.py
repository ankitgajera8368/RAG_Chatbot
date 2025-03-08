import faiss
from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL_NAME
import logging
import config.logging_config

logger = logging.getLogger(__name__)


def retrieve_top_k(query: str, index: faiss.Index, chunks: list[str], top_k=3):
    """
    Given a user query, embed it, perform an FAISS search, and return the top_k chunk texts.
    """
    # Embed the query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search
    distances, indices = index.search(query_embedding, top_k)
    # indices -> shape: (query_count, top_k)
    # distances -> shape: (query_count, top_k)

    # Gather top chunks
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results
