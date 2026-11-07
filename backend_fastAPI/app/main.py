from fastapi.openapi.docs import get_swagger_ui_html

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.api.routes import chat, documents, health
from app.core.config import settings
from app.core.exceptions import AppError
from app.services.vector_store import get_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()  # stop early if keys are missing
    get_embeddings()  # load the embedding model once, at start
    logger.info("Server is ready.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,  # we serve our own dark /docs below
)

# CORS (for the React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers: every error has the same JSON shape
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(part) for part in err["loc"][1:]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Your request data is not valid.",
                "details": details,
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Something went wrong on the server. Please try again.",
            }
        },
    )


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


# serving our own dark /docs below
DARK_CSS = """
<style>
  html { background: #fff; filter: invert(90%) hue-rotate(180deg); }
  img, .microlight { filter: invert(100%) hue-rotate(180deg); }
</style>
"""

@app.get("/docs", include_in_schema=False)
def dark_docs():
    page = get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - Docs")
    html = page.body.decode().replace("</head>", DARK_CSS + "</head>")
    return HTMLResponse(html)
