from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentProcessingOptions(BaseModel):
    """Model for document processing options."""
    chunk_size: int = 500
    chunk_overlap: int = 50

class Document(BaseModel):
    """Model for document data."""
    document_id: Optional[int] = None
    title: str
    content: str
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    document_type: Optional[str] = None

    class Config:
        orm_mode = True


class Chunk(BaseModel):
    """Model for text chunk data."""
    chunk_id: Optional[int] = None
    document_id: int
    content: str
    chunk_order: int
    created_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class QueryRequest(BaseModel):
    """Model for RAG query request."""
    query: str = Field(..., description="The query text to process")
    max_chunks: Optional[int] = Field(5, description="Maximum number of chunks to retrieve")
    temperature: Optional[float] = Field(0.7, description="Temperature for the LLM")
    include_sources: Optional[bool] = Field(True, description="Whether to include sources in response")


class RetrievedChunk(BaseModel):
    """Model for chunks retrieved during the RAG process."""
    chunk_id: int
    document_id: int
    content: str
    document_title: Optional[str] = None
    document_source: Optional[str] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Model for RAG query response."""
    query: str
    response: str
    chunks: Optional[List[RetrievedChunk]] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    title: str
    content: str
    source: Optional[str] = None
    document_type: Optional[str] = None


class DocumentUpdate(BaseModel):
    """Model for updating an existing document."""
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    document_type: Optional[str] = None


class ChunkCreate(BaseModel):
    """Model for creating a new chunk."""
    document_id: int
    content: str
    chunk_order: int


class ChunkUpdate(BaseModel):
    """Model for updating an existing chunk."""
    content: Optional[str] = None
    chunk_order: Optional[int] = None