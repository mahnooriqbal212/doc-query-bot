# Documentation Chatbot API

A RAG backend built with FastAPI. Give it a documentation link (a PDF or a web page, like Express or MongoDB docs), then ask coding questions. Answers come ONLY from that document. If the answer is not in the document, the bot replies exactly: `I don't know`.

## Features

- Load a document from a PDF link or a web page link
- Answers only from the document (no outside knowledge)
- Chat memory per session, so follow-up questions work
- Clear chat memory at any time
- CORS enabled for a React frontend
- Clear error messages in one JSON format

## Tech Stack

- FastAPI, Uvicorn
- LangChain (history-aware retriever + retrieval chain)
- Groq (model: openai/gpt-oss-120b)
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Pinecone (serverless vector database)

## Project Structure

```
langchain_RAG_chatbot/
├── LICENSE
├── README.md
└── backend_fastAPI/
    ├── .env.example
    ├── requirements.txt
    ├── docs/                  # downloaded PDFs
    └── app/
        ├── main.py            # app, CORS, error handlers
        ├── core/              # settings and custom errors
        ├── schemas/           # request and response shapes
        ├── services/          # RAG logic
        └── api/routes/        # endpoints
```

## How It Works

1. Load the document (PyPDFLoader or WebBaseLoader)
2. Split it into chunks (200 tokens, overlap 20)
3. Create embeddings
4. Store them in Pinecone (old vectors are deleted first)
5. Retrieve the top 3 chunks for a question
6. Send the chunks and the question to the LLM with a strict prompt
7. Build the RAG chain
8. Add chat memory
9. Return the answer

## Setup

1. Clone the project and go to the backend folder:

```
cd backend_fastAPI
```

2. Create and activate a virtual environment, then install packages:

```
python -m venv .venv
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your keys:

```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=docs-chatbot
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

4. Run the server:

```
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | /health | Checks that the server runs |
| POST | /documents | Loads and indexes a document from a link |
| POST | /chat | Asks a question |
| DELETE | /chat/{session_id} | Clears the chat memory |

### POST /documents

```
{ "url": "https://expressjs.com/en/guide/routing.html" }
```

### POST /chat

```
{ "session_id": "user-1", "question": "How do I create a route?" }
```

### Error format

```
{ "error": { "code": "NO_DOCUMENT", "message": "No document is loaded yet. Please add a documentation link first." } }
```

## Roadmap

- React.js frontend
- Read a whole documentation site, not only one page

## License

MIT License. See the LICENSE file.