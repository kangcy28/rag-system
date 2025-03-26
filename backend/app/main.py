from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import logging

from app.config import Settings, get_settings
from app.database.connection import get_db
from app.models.models import QueryRequest, QueryResponse, Document, Chunk
from app.services.rag_service import RAGService
from app.routes.api import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API with LangChain",
    description="Retrieval-Augmented Generation API using LangChain and Azure OpenAI",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} in {settings.environment} environment")
    logger.info(f"Using Azure OpenAI: {settings.use_azure}")
    logger.info(f"Database: {settings.db_server}:{settings.db_port}/{settings.db_name}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to RAG API with LangChain",
        "documentation": "/docs", 
        "health": "/health"
    }

@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    try:
        return {
            "status": "healthy",
            "environment": settings.environment,
            "version": "1.0.0",
            "services": {
                "database": "connected",
                "azure_openai": "configured" if settings.use_azure and settings.azure_endpoint else "not_configured",
                "langchain": "enabled"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/config")
async def get_config(settings: Settings = Depends(get_settings)):
    """Get non-sensitive configuration information."""
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "use_azure": settings.use_azure,
        "model_name": settings.model_name,
        "default_chunk_size": settings.default_chunk_size,
        "default_chunk_overlap": settings.default_chunk_overlap
    }