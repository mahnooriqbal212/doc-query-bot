import os
from pathlib import Path
from dotenv import load_dotenv
from app.core.exceptions import ConfigError

# backend_fastAPI/ folder
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = "Documentation Chatbot API"
    APP_VERSION = "1.0.0"

    # Keys (from .env)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

    # Pinecone
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs-chatbot")
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    # Splitter and retriever
    TOKENS_PER_CHUNK = 200
    CHUNK_OVERLAP = 20
    RETRIEVER_K = 3

    # LLM
    LLM_MODEL = "openai/gpt-oss-120b"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500

    # Documents
    DOCS_DIR = BASE_DIR / "docs"
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