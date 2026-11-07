# Documentation Chatbot API

A RAG (Retrieval-Augmented Generation) backend built with FastAPI. Give it documentation, as a PDF link, a web page link (like the Express or MongoDB docs) or a PDF file from your computer, and ask coding questions about it.

Answers come **only** from that document, and every answer shows the pages it came from. If the answer is not in the document, the bot replies `I don't know`.

## Features

**Documents**
- Load a document from a **PDF link**, a **web page link**, or a **PDF upload**
- **Many documents at once**: each document has its own `document_id` and its own space in Pinecone (a namespace), so users don't overwrite each other
- List, view and delete documents. The list is kept in `data/documents.json`, so it survives a server restart
- Safety checks: links to private or local addresses are blocked, PDFs are limited to 25 MB, and files that only pretend to be PDFs are rejected

**Answers**
- Answers come only from the document's context (no outside knowledge)
- **Sources** come with every answer: link or file name, page number, a short text snippet and a similarity score
- **Relevance check**: if no part of the document is related to the question, the bot answers `I don't know` without calling the LLM. This is faster, free, and gives no made-up answers
- Temperature 0, so answers stay close to the text
- Markdown answers with code blocks
- **Streaming**: an extra endpoint sends the answer word by word (Server-Sent Events), like ChatGPT

**Chat memory**
- Memory per session, so follow-up questions work ("and what about POST?")
- Only the last 10 messages are sent to the LLM, which keeps it fast and cheap
- Long chats keep only their newest 100 messages. Sessions idle for 60 minutes are removed
- Get a chat's full history again (for example after a page refresh), or clear it

**API quality**
- Every error has the same JSON shape with a clear message
- CORS is enabled for a React frontend
- Dark mode Swagger UI at `/docs`
- Automated tests (no API keys or internet needed)

## Tech Stack

- **FastAPI** and **Uvicorn**: web API
- **LangChain**: prompts, LLM calls, document loaders and the text splitter
- **Groq**: LLM (`openai/gpt-oss-120b`)
- **HuggingFace** embeddings: `all-MiniLM-L6-v2` (384 dimensions, runs on your computer)
- **Pinecone**: serverless vector database (one namespace per document)
- **pytest**: tests

## Project Structure

```
langchain_RAG_chatbot/
├── LICENSE
├── README.md
└── backend_fastAPI/
    ├── .env.example
    ├── requirements.txt
    ├── pytest.ini
    ├── data/                     # created automatically on the first document
    │   ├── documents.json        # list of loaded documents
    │   └── temp/                 # temporary PDF files (deleted after reading)
    ├── tests/                    # automated tests
    └── app/
        ├── main.py               # app, CORS, error handlers, dark /docs
        ├── core/
        │   ├── config.py         # all settings in one place
        │   └── exceptions.py     # our error types (code + status + message)
        ├── schemas/
        │   ├── document.py       # document request and response shapes
        │   └── chat.py           # chat request and response shapes, Source
        ├── services/
        │   ├── document_loader.py    # 1. load a PDF link, web page or upload
        │   ├── vector_store.py       # 2-5. split, embed, store, search (Pinecone)
        │   ├── document_registry.py  # list of loaded documents (JSON file)
        │   ├── rag_chain.py          # 6-7. LLM, prompts, question rewriting
        │   ├── chat_memory.py        # 8. sessions and history limits
        │   └── chat_service.py       # 9. answer flow (normal + streaming)
        └── api/routes/
            ├── health.py
            ├── documents.py
            └── chat.py
```

## How It Works

**Adding a document** (`POST /documents` or `POST /documents/upload`)

1. **Load**: `PyPDFLoader` for PDFs (one page each), `WebBaseLoader` for web pages
2. **Split** with `RecursiveCharacterTextSplitter` into chunks of 250 tokens with an overlap of 30. It cuts at paragraphs first, then lines, then words, and keeps the original text exactly as written (upper case and code stay correct). Chunk size is counted with the embedding model's own tokenizer, because the model reads at most 256 tokens
3. **Embed** each chunk into 384 numbers with `all-MiniLM-L6-v2`
4. **Store** the chunks in a new Pinecone **namespace** named after a new `document_id`. Other documents are not touched. If saving fails, the half-saved namespace is deleted

**Asking a question** (`POST /chat` or `POST /chat/stream`)

5. **Rewrite** the question using the chat history, so a follow-up becomes a full question. This step is skipped when there is no history yet
6. **Search** the document's namespace for the 5 most similar chunks
7. **Filter** out chunks with a similarity below `0.3`. If none are left, the answer is `I don't know` and the LLM is not called
8. **Answer** with the LLM, using a strict prompt: only the context, plus the last 10 messages
9. **Save** the question and answer (with sources) in the session's memory and return them

## Setup

**Requirements:** Python 3.13. On Windows, also the latest [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe), which PyTorch needs.

1. Go to the backend folder:

   ```
   cd backend_fastAPI
   ```

2. Create and activate a virtual environment, then install packages:

   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1        # Windows PowerShell
   source .venv/bin/activate         # macOS / Linux
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your keys:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=docs-chatbot
   HF_TOKEN=your_huggingface_token_here      # optional, hides a warning
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   USER_AGENT=DocumentationChatbot/1.0
   ```

   | Key | Where to get it | Required |
   |-----|-----------------|----------|
   | `GROQ_API_KEY` | https://console.groq.com/keys | Yes |
   | `PINECONE_API_KEY` | https://app.pinecone.io (API Keys) | Yes |
   | `HF_TOKEN` | https://huggingface.co/settings/tokens (Read token) | No |

   The Pinecone index is created automatically the first time you add a document.

4. Run the server:

   ```
   uvicorn app.main:app --reload
   ```

   Open http://127.0.0.1:8000/docs to try the API. The first start downloads the embedding model (about 90 MB).

## Running the Tests

```
cd backend_fastAPI
pytest
```

The tests replace Pinecone, Groq and the embedding model with fakes, so they run in about a second, need no API keys and cost nothing. They cover the endpoints, the error format, sources, the relevance check, streaming, history limits, session cleanup, upload checks and the link safety check.

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/health` | Checks that the server is running |
| POST | `/documents` | Loads and indexes a document from a link |
| POST | `/documents/upload` | Loads and indexes an uploaded PDF file |
| GET | `/documents` | Lists all loaded documents (newest first) |
| GET | `/documents/{document_id}` | Shows one document |
| DELETE | `/documents/{document_id}` | Deletes a document, its vectors and its chats |
| POST | `/chat` | Asks a question, returns the full answer with sources |
| POST | `/chat/stream` | Asks a question, streams the answer word by word |
| GET | `/chat/{session_id}` | Returns the chat history |
| DELETE | `/chat/{session_id}` | Clears the chat memory |

### Add a document from a link: `POST /documents`

Request:

```json
{ "url": "https://expressjs.com/en/guide/routing.html" }
```

Response (`201`):

```json
{
  "message": "Document is ready. You can ask questions now.",
  "document_id": "3f2a9c1e8b7d4e6f9a0b1c2d3e4f5a6b",
  "source": "https://expressjs.com/en/guide/routing.html",
  "pages": 1,
  "chunks": 42,
  "created_at": "2026-11-08T10:15:00.000000+00:00"
}
```

Save the `document_id`, because every chat request needs it.

### Upload a PDF: `POST /documents/upload`

Send the file as `multipart/form-data` in a field called `file`. The response is the same as above, and `source` is the file name.

```
curl -F "file=@mongodb-manual.pdf" http://127.0.0.1:8000/documents/upload
```

### Ask a question: `POST /chat`

Request:

```json
{
  "session_id": "user-1",
  "document_id": "3f2a9c1e8b7d4e6f9a0b1c2d3e4f5a6b",
  "question": "How do I create a route that handles POST requests?"
}
```

Response:

```json
{
  "session_id": "user-1",
  "document_id": "3f2a9c1e8b7d4e6f9a0b1c2d3e4f5a6b",
  "answer": "Use `app.post()`:\n```javascript\napp.post('/', (req, res) => {\n  res.send('Got a POST request')\n})\n```",
  "sources": [
    {
      "source": "https://expressjs.com/en/guide/routing.html",
      "page": null,
      "snippet": "The following code is an example of routes that are defined for the GET and the POST methods to the root of the app...",
      "score": 0.71
    }
  ]
}
```

- `page` is the PDF page number (starting at 1), or `null` for web pages.
- `score` is the similarity from 0 to 1. Higher means more related.
- When the answer is `I don't know`, `sources` is empty.
- `session_id` is chosen by the frontend, for example a random id per chat. One session belongs to one document. If you use the same `session_id` with a different document, a new empty chat starts.

### Stream the answer: `POST /chat/stream`

The request body is the same as `/chat`. The response is `text/event-stream` with these events:

| Event | Data | When |
|-------|------|------|
| `token` | `{"text": "..."}` | Many times, one piece of the answer each time |
| `done` | `{"answer": "...", "sources": [...]}` | Once at the end |
| `error` | `{"code": "...", "message": "..."}` | If the AI model fails midway |

Example stream:

```
event: token
data: {"text": "Use"}

event: token
data: {"text": " `app.post()`"}

event: done
data: {"answer": "Use `app.post()` ...", "sources": [...]}
```

Errors found **before** streaming starts, such as an unknown document or bad input, come back as normal JSON errors with a status code.

The browser's `EventSource` only supports GET, so read the stream with `fetch` in React:

```javascript
const response = await fetch("http://127.0.0.1:8000/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id, document_id, question }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const events = buffer.split("\n\n");
  buffer = events.pop(); // keep the unfinished part
  for (const block of events) {
    const event = block.match(/^event: (.*)$/m)[1];
    const data = JSON.parse(block.match(/^data: (.*)$/m)[1]);
    if (event === "token") appendToAnswer(data.text);
    if (event === "done") showSources(data.sources);
    if (event === "error") showError(data.message);
  }
}
```

### Chat history: `GET /chat/{session_id}`

```json
{
  "session_id": "user-1",
  "document_id": "3f2a9c1e8b7d4e6f9a0b1c2d3e4f5a6b",
  "messages": [
    { "role": "user", "content": "How do I create a route?", "sources": [] },
    { "role": "assistant", "content": "Use `app.get()` ...", "sources": [ ... ] }
  ]
}
```

## Errors

Every error has the same shape:

```json
{ "error": { "code": "DOCUMENT_NOT_FOUND", "message": "This document was not found. Please add a documentation link first." } }
```

Validation errors also have a `details` list with the wrong `field` and its `message`.

| Code | Status | Meaning |
|------|--------|---------|
| `VALIDATION_ERROR` | 422 | The request data is not valid (empty question, bad URL, ...) |
| `INVALID_LINK` | 400 | Website not found, or a private/local address |
| `INVALID_FILE` | 400 | The uploaded file is not a `.pdf` |
| `DOCUMENT_LOAD_ERROR` | 422 | The document could not be downloaded or read, or has no text |
| `DOCUMENT_TOO_LARGE` | 413 | The PDF is bigger than 25 MB |
| `DOCUMENT_NOT_FOUND` | 404 | Unknown `document_id` |
| `SESSION_NOT_FOUND` | 404 | Unknown `session_id` |
| `RATE_LIMITED` | 429 | Too many requests to Groq. Wait and try again |
| `LLM_ERROR` | 502 | The AI model failed |
| `VECTOR_STORE_ERROR` | 502 | Pinecone failed (check your key) |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

## Configuration

All settings are in `backend_fastAPI/app/core/config.py`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `TOKENS_PER_CHUNK` / `CHUNK_OVERLAP` | 250 / 30 | Chunk size (the model's limit is 256) |
| `RETRIEVER_K` | 5 | Chunks fetched per question |
| `MIN_RELEVANCE_SCORE` | 0.3 | Chunks below this score are ignored |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `LLM_TEMPERATURE` | 0 | 0 = stick to the text |
| `LLM_MAX_TOKENS` | 1024 | Maximum answer length |
| `LLM_REASONING_EFFORT` | `low` | How long the model "thinks" before answering |
| `HISTORY_MESSAGES_FOR_LLM` | 10 | Messages of history sent to the LLM |
| `MAX_MESSAGES_PER_SESSION` | 100 | Messages kept per chat |
| `SESSION_TTL_MINUTES` | 60 | Idle sessions are removed after this |
| `MAX_PDF_SIZE_MB` | 25 | Largest allowed PDF |

## Limitations

- Chat memory is kept in RAM. It is lost when the server restarts, and it only works with one Uvicorn worker.
- A web page link reads **one page**, not the whole documentation site.
- Scanned PDFs (only images, no text) cannot be read.
- Documents stay in Pinecone until you delete them with `DELETE /documents/{document_id}`.

## Roadmap

- React.js + Tailwind frontend
- Read a whole documentation site (sitemap or crawling)
- Save chat history in a database (SQLite or Redis)
- Rate limiting per user
- Background indexing with progress for large documents
- Docker setup

## License

MIT License. See the LICENSE file.
