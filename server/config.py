import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file if present for the credentials

# store env variables or other config constants
KRUTRIM_CLOUD_API_KEY = os.getenv("KRUTRIM_CLOUD_API_KEY")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Llama-3.3-70B-Instruct")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./resources/faiss/faiss.index")

FAISS_METADATA_PATH = os.getenv(
    "FAISS_METADATA_PATH", "./resources/faiss/faiss.index.metadata.json"
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

DATASET_PATH = (
    r"E:\Docs\AbInBev\Drive_Code\demo_bot_data\demo_bot_data\demo_bot_data\ubuntu-docs"
)

MAX_CHARS = 5000

CHAT_HISTORY = []

WEB_SERVER_IP = os.getenv("WEB_SERVER_IP", "localhost")
WEB_SERVER_PORT = os.getenv("WEB_SERVER_PORT", "7007")
WEB_SERVER_URL = "http://{WEB_SERVER_IP}:{WEB_SERVER_PORT}/"
