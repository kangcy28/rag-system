from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.config import Settings, get_settings
from app.database.connection import get_db
from app.models.models import QueryRequest, QueryResponse, Document, Chunk
from app.services.rag_service import RAGService
from app.routes.api import router as api_router

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API using LangChain and OpenAI",
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

@app.get("/")
async def root():
    return {"message": "Welcome to RAG API. Visit /docs for the API documentation."}

@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    try:
        return {
            "status": "healthy",
            "environment": settings.environment,
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")