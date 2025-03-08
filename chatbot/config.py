import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file if present for the credentials

# store env variables or other config constants

WEB_SERVER_IP = os.getenv("WEB_SERVER_IP", "localhost")
WEB_SERVER_PORT = os.getenv("WEB_SERVER_PORT", "7007")
WEB_SERVER_URL = f"http://{WEB_SERVER_IP}:{WEB_SERVER_PORT}"
