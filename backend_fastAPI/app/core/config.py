import os
from pathlib import Path
from dotenv import load_dotenv
from app.core.exceptions import ConfigError

# backend_fastAPI/ folder
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = "Documentation Chatbot API"
    APP_VERSION = "2.0.0"

    # Keys (from .env)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

    # Pinecone (each document gets its own namespace inside this one index)
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs-chatbot")
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    # Splitter and retriever
    # all-MiniLM-L6-v2 reads at most 256 tokens, so a chunk must not be bigger.
    TOKENS_PER_CHUNK = 250
    CHUNK_OVERLAP = 30
    RETRIEVER_K = 5
    # Cosine similarity (0 to 1). Chunks below this are "not related".
    # If no chunk passes, we answer "I don't know" without calling the LLM.
    MIN_RELEVANCE_SCORE = 0.3

    # LLM
    LLM_MODEL = "openai/gpt-oss-120b"
    LLM_TEMPERATURE = 0  # 0 = no creativity, stick to the context
    LLM_MAX_TOKENS = 1024
    LLM_REASONING_EFFORT = "low"  # gpt-oss thinks less, so more tokens go to the answer
    NOT_FOUND_ANSWER = "I don't know"

    # Chat memory
    HISTORY_MESSAGES_FOR_LLM = 10  # only the last 10 messages are sent to the LLM
    MAX_MESSAGES_PER_SESSION = 100  # older messages are dropped
    SESSION_TTL_MINUTES = 60  # sessions idle for longer are deleted

    # Documents
    DATA_DIR = BASE_DIR / "data"  # created automatically when the first document is added
    DOCUMENTS_FILE = DATA_DIR / "documents.json"  # list of loaded documents
    DOCS_DIR = DATA_DIR / "temp"  # temporary PDF files (deleted right after reading)
    REQUEST_TIMEOUT = 30  # seconds
    MAX_PDF_SIZE_MB = 25

    # CORS (React frontend URLs, separated by commas)
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        ).split(",")
        if origin.strip()
    ]

    def validate(self) -> None:
        missing = [
            name
            for name in ("GROQ_API_KEY", "PINECONE_API_KEY")
            if not getattr(self, name)
        ]
        if missing:
            raise ConfigError(
                f"Missing environment variables: {', '.join(missing)}. "
                "Add them to backend_fastAPI/.env"
            )


settings = Settings()
