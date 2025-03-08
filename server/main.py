from fastapi import FastAPI
from routers import chat
import logging
import config.logging_config

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chatbot Microservice",
        description="A microservice that retrieves relevant docs from FAISS and calls LLM model to generate response.",
        version="1.0.0",
    )
    # Include Routers
    app.include_router(chat.router)

    return app


app = create_app()
