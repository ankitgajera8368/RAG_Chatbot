from fastapi import FastAPI
from routers import chat


def create_app() -> FastAPI:
    app = FastAPI(
        title="Markdown Chatbot Microservice",
        description="A microservice that retrieves relevant docs from FAISS and calls KrutrimCloud LLM.",
        version="1.0.0",
    )
    # Include Routers
    app.include_router(chat.router)

    return app


app = create_app()
