import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


def retrieve_top_k(query: str, index: faiss.Index, chunks: list[str], top_k=3):
    """
    Given a user query, embed it, perform an FAISS search, and return the top_k chunk texts.
    """
    # Embed the query
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    # Gather top chunks
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results


def semantic_search(
    query,
    index,
    chunks,
    top_k=3,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Performs a semantic search on the provided FAISS index, returning the top_k chunks.
    """
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query], convert_to_numpy=True)

    # FAISS search
    D, I = index.search(query_vector, top_k)  # Distances and Indices
    # I -> shape: (query_count, top_k)
    # D -> shape: (query_count, top_k)

    top_results = []
    for i, idx_ in enumerate(I[0]):
        result = {"chunk": chunks[idx_], "distance": float(D[0][i])}
        top_results.append(result)

    return top_results
