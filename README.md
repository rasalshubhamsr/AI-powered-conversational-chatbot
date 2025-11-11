# Conversational Chatbot API

An AI-powered conversational chatbot using FastAPI, LangChain, and OpenAI. It supports conversation with context, sentiment analysis, and entity extraction.

## Features
- `/chat`: Chat with the bot using natural language, with conversation memory and optional context
- `/context`: Get or set global conversation context
- `/sentiment`: Classify input as positive, negative, or neutral
- `/entities`: Extract named entities from the input

## Requirements
- Python 3.10+
- An OpenAI API key available as environment variable `OPENAI_API_KEY`

## Install
```
# Backend deps
pip install -r backend/requirements.txt
```

## Run
```
# From repo root
uvicorn backend.main:app --reload --port 8000
```

## Dependency check
```
python scripts/check_dependencies.py
```
If issues are reported, try:
```
pip install --upgrade -r backend/requirements.txt
```

## Frontend
- A simple HTML tester lives in `frontend/index.html`.
- When you run the backend, open http://localhost:8000/ to load it.

## API Endpoints
1. POST `/chat`
   - Request: `{ "input": "user input" }`
   - Response: `{ "response": "chatbot response" }`

2. GET `/context`
   - Response: `{ "context": "current conversation context" }`

   POST `/context`
   - Request: `{ "context": "new conversation context" }`
   - Response: `{ "message": "Context updated successfully" }`

3. POST `/sentiment`
   - Request: `{ "input": "user input" }`
   - Response: `{ "sentiment": "positive|negative|neutral" }`

4. POST `/entities`
   - Request: `{ "input": "user input" }`
   - Response: `{ "entities": ["entity1", "entity2", ...] }`

## Notes
- This sample keeps a single global conversation memory for simplicity. In production, use per-user/per-session memories (e.g., via headers or tokens) and persistent storage.
- All LLM calls are powered by OpenAI models via LangChain (`gpt-4o-mini` by default). You can change the model with the `OPENAI_MODEL` env var.
