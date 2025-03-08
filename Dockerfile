# Use a lightweight Python base
FROM python:3.12-slim

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY chatbot ./chatbot
COPY server ./server
COPY .env .env

# Expose the port on which FastAPI will run
EXPOSE 7007

WORKDIR ./server

# Define a default command to run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7007"]
