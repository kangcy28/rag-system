from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.database.connection import get_db
from app.models.models import (
    QueryRequest, QueryResponse, Document, Chunk,
    DocumentCreate, DocumentUpdate, ChunkCreate,
    DocumentProcessingOptions  # Import the model from models.py
)
from app.database.repository import DocumentRepository, ChunkRepository
from app.services.rag_service import RAGService

router = APIRouter()

# RAG Query Endpoint
@router.post("/query", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a query using RAG.
    """
    try:
        rag_service = RAGService(db)
        response = rag_service.process_query(query_request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

# Document Endpoints
@router.get("/documents", response_model=List[Dict[str, Any]])
async def get_documents(
    skip: int = 0, 
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get all documents with pagination.
    """
    return DocumentRepository.get_documents(db, skip=skip, limit=limit)

@router.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific document by ID.
    """
    document = DocumentRepository.get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return document

@router.post("/documents", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new document.
    """
    doc_model = Document(
        title=document.title,
        content=document.content,
        source=document.source,
        document_type=document.document_type
    )
    return DocumentRepository.create_document(db, doc_model)

@router.put("/documents/{document_id}", response_model=Dict[str, Any])
async def update_document(
    document_id: int,
    document: DocumentUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing document.
    """
    doc_dict = document.dict(exclude_unset=True)
    updated_doc = DocumentRepository.update_document(db, document_id, doc_dict)
    if not updated_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found or no fields to update"
        )
    return updated_doc

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its chunks.
    """
    success = DocumentRepository.delete_document(db, document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return None

# Chunk Endpoints
@router.get("/documents/{document_id}/chunks", response_model=List[Dict[str, Any]])
async def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a document.
    """
    # First check if document exists
    document = DocumentRepository.get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    return ChunkRepository.get_chunks_by_document(db, document_id)

@router.post("/chunks", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_chunk(
    chunk: ChunkCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new chunk.
    """
    # Check if document exists
    document = DocumentRepository.get_document_by_id(db, chunk.document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {chunk.document_id} not found"
        )
    
    chunk_model = Chunk(
        document_id=chunk.document_id,
        content=chunk.content,
        chunk_order=chunk.chunk_order
    )
    return ChunkRepository.create_chunk(db, chunk_model)

@router.post("/documents/{document_id}/process", response_model=None)
async def process_document(
    document_id: int,
    options: DocumentProcessingOptions,
    db: Session = Depends(get_db)
):
    """
    Process a document using LangChain text splitters to create chunks.
    
    Args:
        document_id: The document ID to process
        options: Document processing options (chunk size and overlap)
        
    Returns:
        Dict with processing results
    """
    # Check if document exists
    document = DocumentRepository.get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    # Delete existing chunks for this document
    # This ensures we don't have duplicate chunks when reprocessing
    delete_query = """
    DELETE FROM Chunks
    WHERE document_id = :document_id
    """
    db.execute(delete_query, {"document_id": document_id})
    db.commit()
    
    # Process document with LangChain
    rag_service = RAGService(db)
    num_chunks = rag_service.chunk_document(
        document_id, 
        chunk_size=options.chunk_size, 
        overlap=options.chunk_overlap
    )
    
    return {
        "document_id": document_id,
        "document_title": document["title"],
        "chunks_created": num_chunks,
        "chunk_size": options.chunk_size,
        "chunk_overlap": options.chunk_overlap,
        "status": "success",
        "message": f"Successfully processed document and created {num_chunks} chunks"
    }