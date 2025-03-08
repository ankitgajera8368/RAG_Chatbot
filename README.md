# Ubuntu Setup Helper Bot


This repository contains a Retrieval-Augmented Generation (RAG) chatbot built with FastAPI and various NLP components. The chatbot answers user queries by retrieving relevant context from local documents (via a FAISS index) and then generating a response using a Large Language Model (LLM).


## Features

- **Retrieval**: Utilizes FAISS for efficient similarity search over embedded documents.  
- **Augmented Generation**: Automatically injects relevant chunks into the LLM prompt to provide richer context.  
- **FastAPI**: Provides a RESTful API for chat interactions, returning streaming responses.
- **Conversation History**: Chat service maintains the conversation history and respond to queries based on previous conversation

## Output

### Start a conversation with bot
![Chatbot Initial Conversation](snapshots\Chatbot_Initial_Conversation.png?raw=true "Chatbot Initial Conversation")

### Service maintains the conversation history and respond to queries based on previous conversation
![Chatbot Continue Conversation](snapshots\Chatbot_Continue_Conversation.png?raw=true "Chatbot Continue Conversation")

### Also providing the references to cross validate the credebility of the response
![References](snapshots\References.png?raw=true "References")

### Swagger Doc Output for the microservice
![Swagger Doc Output](snapshots\SwaggerDoc_Output.png?raw=true "Swagger Doc Output")