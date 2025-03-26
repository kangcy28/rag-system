import time
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.services.langchain_service import LangChainService
from app.database.repository import ChunkRepository, DocumentRepository
from app.models.models import QueryRequest, QueryResponse, RetrievedChunk

class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(self, db: Session):
        """Initialize the RAG service."""
        self.db = db
        self.langchain_service = LangChainService(db)
    
    def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a query using the RAG approach with LangChain.
        
        Args:
            query_request (QueryRequest): The query request object
            
        Returns:
            QueryResponse: The response including generated text and retrieved chunks
        """
        start_time = time.time()
        
        # Retrieve relevant chunks using LangChain embeddings
        retrieved_chunks = self.langchain_service.retrieve_chunks(
            query_request.query, 
            max_chunks=query_request.max_chunks
        )
        
        # Generate response using LangChain with context
        response_data = self.langchain_service.generate_response(
            query_request.query, 
            context_chunks=retrieved_chunks
        )
        
        # Format retrieved chunks for response
        formatted_chunks = []
        for chunk in retrieved_chunks:
            formatted_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    document_title=chunk.get("document_title"),
                    document_source=chunk.get("document_source"),
                    relevance_score=chunk.get("relevance_score")
                )
            )
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            "model": response_data.get("model"),
            "chunks_retrieved": len(retrieved_chunks)
        }
        
        # Save query and response to database
        ChunkRepository.save_query(
            self.db,
            query_request.query,
            response_data["response"],
            metadata=metadata
        )
        
        # Create response object
        response = QueryResponse(
            query=query_request.query,
            response=response_data["response"],
            chunks=formatted_chunks if query_request.include_sources else None,
            processing_time=processing_time,
            metadata=metadata
        )
        
        return response
    
    def chunk_document(self, document_id: int, chunk_size: int = 500, overlap: int = 50) -> int:
        """
        Chunk a document using LangChain text splitters and store the chunks in the database.
        
        Args:
            document_id (int): The document ID to chunk
            chunk_size (int): The size of each chunk in characters
            overlap (int): The overlap between chunks in characters
            
        Returns:
            int: The number of chunks created
        """
        # Use LangChain service to process document
        return self.langchain_service.process_document(
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )