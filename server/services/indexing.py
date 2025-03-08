import json
import os
import re
from typing import Dict, List
import numpy as np
import yaml
import os
import faiss
from sentence_transformers import SentenceTransformer
import logging
import config.logging_config

logger = logging.getLogger(__name__)


def load_all_markdown_files(folder_path):
    """
    Recursively loads all markdown (.md) files under folder_path,
    returning a list of (absolute_filepath, file_content).
    """
    md_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".md"):
                abs_path = os.path.join(root, filename)
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                md_files.append((abs_path.replace(folder_path, ""), content))
    return md_files


def extract_front_matter(content):
    """
    Extracts front matter in YAML format if it exists at the top of the file.
    Returns a tuple: (metadata_dict, remaining_content).

    Front matter is defined as text between the first two '---' lines:
        ---
        key: value
        ---
    If no front matter is found, returns ({}, content).
    """
    # Regex to capture text between two lines of ---
    front_matter_pattern = r"(?s)^---\n(.*?)\n---\n(.*)"
    match = re.match(front_matter_pattern, content)
    if not match:
        return {}, content  # No front matter

    fm_str, remaining = match.groups()
    try:
        metadata = yaml.safe_load(fm_str)
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, remaining.strip()


def split_markdown_by_headings(content):
    """
    Splits markdown content by top-level headings (#, ##, ###, etc.).
    Returns a list of (heading_text, section_text).
    heading_text might be None if text appears before the first heading.
    """
    # Regex to match headings at the start of a line
    pattern = r"(^#{1,6}\s.*)"
    # Split while capturing the headings
    splits = re.split(pattern, content, flags=re.MULTILINE)

    sections = []
    current_heading = None
    current_content = []

    for part in splits:
        part = part.strip()
        if re.match(r"^#{1,6}\s.*", part):
            # This is a heading
            # If there's existing content, push it first
            if current_heading or current_content:
                combined = "\n".join(current_content).strip()
                if combined:
                    sections.append((current_heading, combined))
            current_heading = part
            current_content = []
        else:
            # Regular text
            if part:
                current_content.append(part)

    # Last accumulated content
    if current_heading or current_content:
        combined = "\n".join(current_content).strip()
        if combined:
            sections.append((current_heading, combined))

    return sections


def chunk_section_text(section_text, max_chars=2000):
    """
    Breaks down a single section's text into sub-chunks if it exceeds max_chars.
    Returns a list of sub-chunk strings.
    """
    chunks = []
    start_idx = 0
    while start_idx < len(section_text):
        end_idx = start_idx + max_chars
        chunk_str = section_text[start_idx:end_idx]
        chunks.append(chunk_str)
        start_idx = end_idx
    return chunks


def chunk_markdown_content(content, max_chars=2000):
    """
    1. Split content by headings -> list of (heading, text).
    2. If text is larger than max_chars, chunk it.
    3. Label chunk headings with (part X).
    4. Return list of dicts: [{"heading": ..., "content": ...}, ...]
    """
    sections = split_markdown_by_headings(content)
    all_chunks = []

    for heading, sec_text in sections:
        if not heading:
            heading_label = "No Heading"
        else:
            heading_label = heading

        if len(sec_text) <= max_chars:
            # Single chunk
            all_chunks.append({"heading": heading_label, "content": sec_text.strip()})
        else:
            # Split into sub-chunks
            sub_chunks = chunk_section_text(sec_text, max_chars)
            for i, sub in enumerate(sub_chunks, start=1):
                part_heading = f"{heading_label} (part {i})"
                all_chunks.append({"heading": part_heading, "content": sub.strip()})

    return all_chunks


def build_chunks_for_folder(folder_path, max_chars=2000):
    """
    Main function that:
      1. Loads all markdown files.
      2. Extracts front matter from each file.
      3. Splits content by headings, and further chunks if needed.
      4. Returns a list of chunk records with metadata.
    """
    md_files = load_all_markdown_files(folder_path)
    chunk_records = []  # We'll collect each chunk with metadata

    for file_path, content in md_files:

        # 1) Extract front matter
        fm, md_body = extract_front_matter(content)

        # 2) Chunk the main markdown body
        file_chunks = chunk_markdown_content(md_body, max_chars=max_chars)

        # 3) For each chunk, store relevant metadata
        for chunk_obj in file_chunks:
            record = {
                "filepath": file_path,
                "metadata": fm,  # front matter metadata dict
                "heading": chunk_obj["heading"],
                "content": chunk_obj["content"],
            }
            chunk_records.append(record)

    return chunk_records


# ------------------------------
# Embedding + FAISS Index
# ------------------------------


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Given a 2D np array of shape (num_chunks, embedding_dimension),
    create and return a Flat L2 FAISS index.
    """
    d = embeddings.shape[1]  # dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def store_index_on_disk(index: faiss.Index, index_filepath: str):
    """
    Writes a FAISS index to disk.
    """
    faiss.write_index(index, index_filepath)


def load_index_from_disk(index_filepath: str) -> faiss.Index:
    """
    Reads and returns a FAISS index from disk.
    """
    return faiss.read_index(index_filepath)


def embed_chunks(
    chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Takes a list of chunk records (with 'content') and returns embeddings as an np.ndarray.
    The order of embeddings matches the order of chunks.
    """

    model = SentenceTransformer(model_name)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


# -------------------------------------------------------
#  Check if index exists, else build
# -------------------------------------------------------


def build_or_load_index(
    folder_path: str,
    index_path: str,
    metadata_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars: int = 2000,
):
    """
    Checks if the FAISS index file at `index_path` exists.
    If it does, load the index + metadata JSON.
    If not, build from scratch, embed, store index + metadata.
    Returns (index, chunk_metadata).
    """
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        # Load existing index + metadata
        logger.info("Loading existing FAISS index and metadata from disk...")
        index = load_index_from_disk(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info("Loading complete of existing FAISS index and metadata from disk")
        return index, metadata
    else:
        # Build from scratch
        logger.info("FAISS index not found. Building from scratch...")

        logger.info("Step 1: Building chunks...")
        all_chunks = build_chunks_for_folder(folder_path, max_chars=max_chars)
        logger.info(f"Total chunks: {len(all_chunks)}")

        logger.info("Step 2: Embedding chunks...")
        embeddings = embed_chunks(all_chunks, model_name=model_name)

        logger.info("Step 3: Creating FAISS index...")
        index = create_faiss_index(embeddings)

        logger.info(f"Step 4: Storing index to disk: {index_path}")
        store_index_on_disk(index, index_path)

        logger.info(f"Storing metadata to disk: {metadata_path}")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"FAISS index built from scratch and store at {metadata_path}")
        return index, all_chunks
